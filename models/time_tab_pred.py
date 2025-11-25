import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodels.latent_hash_embedding import LatentHashEmbedding
from .submodels.latent_time_embedding import StructuredLatentTimeEmbedding


class RetailTabPred(nn.Module):

    def __init__(self, model_config, env_config):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        
        self.latent_hashemb = LatentHashEmbedding(model_config, env_config)
        self.latent_timeemb = StructuredLatentTimeEmbedding(model_config, env_config)

        input_dim = (self.model_config['num_discrete']+1)*self.model_config['latent_dim'] + self.model_config['num_continuous']
        self.fc = nn.Linear(input_dim, 1, bias=True)

    def forward(self, cate, cat_incre, obs, target, time, cat_nograd=False, sample_size=(10,)):
        q_item, q_itemw, p_item, p_itemw = self.latent_hashemb.q_E_and_p_E()

        if cat_nograd:
            with torch.no_grad():
                _, _, item_emb, _, _ = self.latent_hashemb(cate, sample_size=())
        else:
            _, _, item_emb, _, _ = self.latent_hashemb(cate, sample_size=())
        # item_emb  shape(sample_size, minibatch_size, num_discrete, latent_dim)
        
        _, _, item_emb_incre, _, _ = self.latent_hashemb(cat_incre, sample_size=())
        item_emb = torch.cat([item_emb, item_emb_incre], -2)

        item_emb = item_emb.reshape(
            *item_emb.shape[:2],
            self.model_config['num_discrete']*self.model_config['latent_dim'])

        time_sample, q_time, p_time = self.latent_timeemb(item_emb, obs, target, time, sample_size=())
        
        # obs shape(minibatch_size, num_continuous)
        x = torch.cat([item_emb, time_sample, obs], -1)
        x = self.fc(x)
        rate = torch.exp(x).squeeze()
        return rate, q_item, q_itemw, q_time, p_item, p_itemw, p_time