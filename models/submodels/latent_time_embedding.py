import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.poisson import Poisson
from torch.distributions.geometric import Geometric
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta


class StructuredLatentTimeEmbedding(nn.Module):
    def __init__(self, model_config, env_config):
        super(StructuredLatentTimeEmbedding, self).__init__()
        self.device = torch.device(model_config['device'])
        self.model_config = model_config
        self.env_config = env_config
        
        self.input_size = (self.model_config['num_discrete']*self.model_config['latent_dim'] + self.model_config['num_continuous'] + 1 + 1)  # target + time
        self.time_emb = nn.GRU(self.input_size,
                               self.model_config['latent_dim'],
                               batch_first=True)
        self.time_emb_map = nn.Linear(self.model_config['latent_dim'], 2*self.model_config['latent_dim'])  # mu and logstd for amortized variational inference
        
        self.transition = nn.Linear(self.model_config['latent_dim'], 2*self.model_config['latent_dim'])
        self.time_transition = nn.Linear(1, 2*self.model_config['latent_dim'])

    def forward(self, cat_emb, obs, target, time, sample_size=(10,)):
        '''
        Arguments:
          obs -- shape(minibatch_size, L, input_size)
          
        Returns:
          post_sample -- shape(sample_size, minibatch_size, L, latent_dim)
          q_time -- shape(minibatch_size, L, latent_dim)
          p_time -- shape(sample_size, minibatch_size, L, latent_dim)
        '''
        obs_tot = torch.cat([cat_emb, obs, target.unsqueeze(-1), time], -1)
        # posterior
        time_emb, _ = self.time_emb(obs_tot)  # default t_0, output shape(minibatch_size, L, latent_dim)
        time_emb_stats = self.time_emb_map(time_emb)  # shape(minibatch_size, L, 2*latent_dim)
        mu_time_emb = time_emb_stats[...,:self.model_config['latent_dim']]  # shape(minibatch_size, L, latent_dim)
        logsigma_time_emb = time_emb_stats[...,self.model_config['latent_dim']:]  # shape(minibatch_size, L, latent_dim)
        q_time = Normal(mu_time_emb, torch.exp(logsigma_time_emb)+1e-8)
        post_sample = q_time.rsample(sample_size)  # shape(sample_size, minibatch_size, L, latent_dim)
        
        # prior
        param_prior_1 = self.transition(post_sample[...,:-1,:])  # shape(sample_size, minibatch_size, L-1, 2*latent_dim)
        param_prior_0 = torch.zeros(*param_prior_1.shape[:-2], 1, 2*self.model_config['latent_dim']).to(self.device)  # initial prior as standard gaussian
        param_prior = torch.cat([param_prior_0, param_prior_1], dim=-2)  # shape(sample_size, minibatch_size, L, 2*latent_dim)
        mu_prior = param_prior[...,:self.model_config['latent_dim']]
        logstd_prior = param_prior[...,self.model_config['latent_dim']:]
        p_time = Normal(mu_prior, torch.exp(logstd_prior)+1e-8)  # shape(sample_size, minibatch_size, L, latent_dim)
        
        return post_sample, q_time, p_time
    
    # def sample(self, time, steps=1, sample_size=()):
    #     for i in range(steps-1):
    
    # def cond_sample(self, ini_emb, time, steps=1, sample_size=()):
    #     for i in range(1,steps):
