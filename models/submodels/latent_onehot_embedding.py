import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.poisson import Poisson
from torch.distributions.geometric import Geometric
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta
    

class LatentOneHotEmbedding(nn.Module):
    def __init__(self, model_config, env_config):
        super(LatentOneHotEmbedding, self).__init__()
        self.device = torch.device(model_config['device'])
        self.model_config = model_config
        self.env_config = env_config
        
        # variational parameters
        self.mu_emb = nn.Embedding(self.model_config['num_bucket'],
                                   self.model_config['latent_dim'])
        self.logsigma_emb = nn.Embedding(self.model_config['num_bucket'],
                                         self.model_config['latent_dim'])
        nn.init.constant_(self.logsigma_emb.weight, -10)  # initiated small, otherwise stuck in local optimum
        
        # prior distribution
        zero_emb = torch.zeros(self.model_config['num_bucket'],
                            self.model_config['latent_dim']).to(self.device)
        self.prior_mu_emb = nn.Embedding.from_pretrained(zero_emb)
        self.prior_logsigma_emb = nn.Embedding.from_pretrained(zero_emb)
        
    def fill_prior(self, prior_mu_emb, prior_logsigma_emb):
        # embeddings
        self.prior_mu_emb = nn.Embedding.from_pretrained(prior_mu_emb.detach().clone().to(self.device))
        self.prior_logsigma_emb = nn.Embedding.from_pretrained(prior_logsigma_emb.detach().clone().to(self.device))
    
    def q_emb(self, idx):
        mu_emb = self.mu_emb(idx)
        logstd_emb = self.logsigma_emb(idx)
        return Normal(mu_emb, torch.exp(logstd_emb)+1e-8)
    
    def p_emb(self, idx):
        mu_emb = self.prior_mu_emb(idx)
        logstd_emb = self.prior_logsigma_emb(idx)
        return Normal(mu_emb, torch.exp(logstd_emb)+1e-8)

    def q_E_and_p_E(self):
        q_emb = Normal(self.mu_emb.weight, 
                       torch.exp(self.logsigma_emb.weight)+1e-8)
        p_emb = Normal(self.prior_mu_emb.weight,
                       torch.exp(self.prior_logsigma_emb.weight)+1e-8)
        return q_emb, p_emb
        
    def forward(self, raw_idx, sample_size=(10,)):
        '''
        Returns:
          hash_id: emb_hash_id (*raw_idx.shape, num_hash), or weight_hash_id shape(*raw_idx.shape)
          embedding: shape(sample_size, *raw_idx.shape, latent_dim)
        '''
        q_latentemb = self.q_emb(raw_idx)  # shape(*raw_idx.shape, num_hash, latent_dim)
        emb_samples = q_latentemb.rsample(sample_size)  # shape(sample_size, B, ..., K, latent_dim) or shape(sample_size, *raw_idx.shape, num_hash, latent_dim)
        
        return emb_samples