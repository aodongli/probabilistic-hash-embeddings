import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodels.latent_hash_embedding import LatentHashEmbedding


class MovieLensNCF_MLP(nn.Module):

    def __init__(self, model_config, env_config):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        
        self.latent_hashemb = LatentHashEmbedding(model_config, env_config)

        input_dim = self.model_config['num_discrete']*self.model_config['latent_dim']
        self.fc = nn.Linear(input_dim, 1, bias=True)  # try heterogeneous regression

    def forward(self, cate, genre_coding, sample_size=(10,)):
        q_item, q_itemw, p_item, p_itemw = self.latent_hashemb.q_E_and_p_E()
        _, _, item_emb, _, _ = self.latent_hashemb(cate, sample_size=sample_size)

        item_emb = item_emb.reshape(
            *item_emb.shape[:2],
            self.model_config['num_discrete']*self.model_config['latent_dim'])
        
        y = self.fc(item_emb).squeeze(-1)
        return y, q_item, q_itemw, p_item, p_itemw


class MovieLensNCF_MLP2(nn.Module):

    def __init__(self, model_config, env_config):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        
        self.latent_hashemb = LatentHashEmbedding(model_config, env_config)

        input_dim = self.model_config['num_discrete']*self.model_config['latent_dim']
        self.fc = nn.Sequential(
            nn.Linear(input_dim, self.model_config['nn_dim'], bias=True),
            nn.Tanh(),
            nn.Linear(self.model_config['nn_dim'], 1, bias=True),
        )

    def forward(self, cate, genre_coding, sample_size=(10,)):
        q_item, q_itemw, p_item, p_itemw = self.latent_hashemb.q_E_and_p_E()
        _, _, item_emb, _, _ = self.latent_hashemb(cate, sample_size=sample_size)

        item_emb = item_emb.reshape(
            *item_emb.shape[:2],
            self.model_config['num_discrete']*self.model_config['latent_dim'])
        
        y = self.fc(item_emb).squeeze(-1)
        return y, q_item, q_itemw, p_item, p_itemw


class MovieLensNCF_MLP2_AugInput(nn.Module):

    def __init__(self, model_config, env_config):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        
        self.latent_hashemb = LatentHashEmbedding(model_config, env_config)

        input_dim = (self.model_config['num_discrete']*self.model_config['latent_dim'] +  # concatenation
            self.model_config['latent_dim'] +  # pointwise multiplication
            self.model_config['num_genre']  # number of movie genres
        )
        self.fc = nn.Sequential(
            nn.Linear(input_dim, self.model_config['nn_dim'], bias=True),
            nn.Tanh(),
            nn.Linear(self.model_config['nn_dim'], 1, bias=True),
        )

    def forward(self, cate, genre_coding, sample_size=(10,)):
        q_item, q_itemw, p_item, p_itemw = self.latent_hashemb.q_E_and_p_E()
        _, _, item_emb, _, _ = self.latent_hashemb(cate, sample_size=sample_size)

        item_emb_cat = item_emb.reshape(
            *item_emb.shape[:2],
            self.model_config['num_discrete']*self.model_config['latent_dim'])
        item_emb_prod = item_emb[...,0,:] * item_emb[...,1,:]
        genre_coding = genre_coding.expand([item_emb_cat.shape[0]] + list(genre_coding.shape))
        x = torch.cat([item_emb_cat, item_emb_prod, genre_coding], -1)

        y = self.fc(x).squeeze(-1)
        return y, q_item, q_itemw, p_item, p_itemw
