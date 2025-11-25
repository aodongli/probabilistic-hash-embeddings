import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.poisson import Poisson
from torch.distributions.geometric import Geometric
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta

from .universal_hash import HashFamily  # shared hash embeddings for all string 
                                         # fields for simplicity
    

class LatentHashEmbedding(nn.Module):
    def __init__(self, model_config, env_config):
        super(LatentHashEmbedding, self).__init__()
        self.device = torch.device(model_config['device'])
        self.model_config = model_config
        self.env_config = env_config
        
        hashfamily = HashFamily(self.model_config['num_bucket'])  # universal hash
        self.hash_funcs = hashfamily.draw_hashes(self.model_config['num_hash'])
        
        # variational parameters
        #  shared embeddings
        self.mu_emb = nn.Embedding(self.model_config['num_bucket'],
                                   self.model_config['latent_dim'])
        self.logsigma_emb = nn.Embedding(self.model_config['num_bucket'],
                                         self.model_config['latent_dim'])
        nn.init.constant_(self.logsigma_emb.weight, -10)  # initiated small, otherwise stuck in local optimum
        
        #  weights
        self.mu_weight = nn.Embedding(self.model_config['num_weight'],
                                      self.model_config['num_hash'])
        self.logsigma_weight = nn.Embedding(self.model_config['num_weight'],
                                      self.model_config['num_hash'])
        nn.init.constant_(self.logsigma_weight.weight, -10)  # initiated small, otherwise stuck in local optimum
        
        # prior distribution
        #  shared embeddings
        zero_emb = torch.zeros(self.model_config['num_bucket'],
                            self.model_config['latent_dim']).to(self.device)
        self.prior_mu_emb = nn.Embedding.from_pretrained(zero_emb)
        self.prior_logsigma_emb = nn.Embedding.from_pretrained(zero_emb)
        #  weights
        zero_weight = torch.zeros(self.model_config['num_weight'],
                               self.model_config['num_hash']).to(self.device)
        self.prior_mu_weight = nn.Embedding.from_pretrained(zero_weight)
        self.prior_logsigma_weight = nn.Embedding.from_pretrained(zero_weight)
        
    def fill_prior(self, prior_mu_emb, prior_logsigma_emb, prior_mu_weight, prior_logsigma_weight):
        #  shared embeddings
        self.prior_mu_emb = nn.Embedding.from_pretrained(prior_mu_emb.detach().clone().to(self.device))
        self.prior_logsigma_emb = nn.Embedding.from_pretrained(prior_logsigma_emb.detach().clone().to(self.device))
        #  weights
        self.prior_mu_weight = nn.Embedding.from_pretrained(prior_mu_weight.detach().clone().to(self.device))
        self.prior_logsigma_weight = nn.Embedding.from_pretrained(prior_logsigma_weight.detach().clone().to(self.device))
        
    def emb_hash_val(self, raw_idx):
        # `raw_idx` indicates the MD5 results of the string text
        return torch.stack([f(raw_idx) for f in self.hash_funcs], dim=-1)
    
    def weight_hash_val(self, raw_idx):
        return raw_idx % self.model_config['num_weight']  # hash function
    
    def q_weight(self, hash_idx):
        mu_weight = self.mu_weight(hash_idx)
        logsigma_weight = self.logsigma_weight(hash_idx)
        return Normal(mu_weight, torch.exp(logsigma_weight)+1e-8)
    
    def q_hashemb(self, hash_idx):
        mu_emb = self.mu_emb(hash_idx)
        logstd_emb = self.logsigma_emb(hash_idx)
        return Normal(mu_emb, torch.exp(logstd_emb)+1e-8)
    
    def p_hashemb(self, hash_idx):
        mu_emb = self.prior_mu_emb(hash_idx)
        logstd_emb = self.prior_logsigma_emb(hash_idx)
        return Normal(mu_emb, torch.exp(logstd_emb)+1e-8)
    
    def p_weight(self, hash_idx):
        mu_weight = self.prior_mu_weight(hash_idx)
        logsigma_weight = self.prior_logsigma_weight(hash_idx)
        return Normal(mu_weight, torch.exp(logsigma_weight)+1e-8)

    def q_E_and_p_E(self):
        q_emb = Normal(self.mu_emb.weight, 
                       torch.exp(self.logsigma_emb.weight)+1e-8)
        q_weight = Normal(self.mu_weight.weight, 
                          torch.exp(self.logsigma_weight.weight)+1e-8)
        p_emb = Normal(self.prior_mu_emb.weight,
                       torch.exp(self.prior_logsigma_emb.weight)+1e-8)
        p_weight = Normal(self.prior_mu_weight.weight,
                          torch.exp(self.prior_logsigma_weight.weight)+1e-8)
        return q_emb, q_weight, p_emb, p_weight
        
    def forward(self, raw_idx, sample_size=(10,)):
        '''
        Returns:
          hash_id: emb_hash_id (*raw_idx.shape, num_hash), or weight_hash_id shape(*raw_idx.shape)
          embedding: shape(sample_size, *raw_idx.shape, latent_dim)
        '''
        emb_hash_id = self.emb_hash_val(raw_idx)  # shape(*raw_idx.shape, num_hash)
        weight_hash_id = self.weight_hash_val(raw_idx)  # raw_idx.shape
        q_latentemb = self.q_hashemb(emb_hash_id)  # shape(*raw_idx.shape, num_hash, latent_dim)
        q_latentw = self.q_weight(weight_hash_id)  # shape(*raw_idx.shape, num_hash)
        emb_samples = q_latentemb.rsample(sample_size)  # shape(sample_size, B, ..., K, latent_dim) or shape(sample_size, *raw_idx.shape, num_hash, latent_dim)
        weight_samples = q_latentw.rsample(sample_size)  # shape(sample_size, B, ..., K), shape(sample_size, *raw_idx.shape, num_hash)
        # aggregate: weighted sum
        weighted_emb = torch.sum(emb_samples * weight_samples.unsqueeze(-1), -2)
        # hash id
        hash_id = emb_hash_id # weight_hash_id  # emb_hash_id
        
        return emb_hash_id, weight_hash_id, weighted_emb, emb_samples, weight_samples
    
    def sample(self, emb_hash_id, weight_hash_id, sample_size=()):
        q_latentemb = self.q_hashemb(emb_hash_id)  # shape(*raw_idx.shape, num_hash, latent_dim)
        q_latentw = self.q_weight(weight_hash_id)  # shape(*raw_idx.shape, num_hash)
        emb_samples = q_latentemb.sample(sample_size)  # shape(sample_size, B, ..., K, latent_dim) or shape(sample_size, *raw_idx.shape, num_hash, latent_dim)
        weight_samples = q_latentw.sample(sample_size)  # shape(sample_size, B, ..., K), shape(sample_size, *raw_idx.shape, num_hash)
        
        weighted_emb = torch.sum(emb_samples * weight_samples.unsqueeze(-1), -2)
        return weighted_emb, q_latentemb, q_latentw