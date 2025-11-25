import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodels.latent_onehot_embedding import LatentOneHotEmbedding


class AdultTabBinCls(nn.Module):

    def __init__(self, model_config, env_config):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        
        self.latent_onehotemb = LatentOneHotEmbedding(model_config, env_config)

        input_dim = self.model_config['num_discrete']*self.model_config['latent_dim'] + self.model_config['num_continuous']
        self.fc = nn.Linear(input_dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, cate, cat_incre, obs, cat_nograd=False, sample_size=(10,)):
        q_item, p_item = self.latent_onehotemb.q_E_and_p_E()

        if cat_nograd:
            with torch.no_grad():
                item_emb = self.latent_onehotemb(cate, sample_size=sample_size)
        else:
            item_emb = self.latent_onehotemb(cate, sample_size=sample_size)
        # item_emb  shape(sample_size, minibatch_size, num_discrete, latent_dim)
        
        item_emb_incre = self.latent_onehotemb(cat_incre, sample_size=sample_size)
        item_emb = torch.cat([item_emb, item_emb_incre], -2)

        item_emb = item_emb.reshape(
            *item_emb.shape[:2],
            self.model_config['num_discrete']*self.model_config['latent_dim'])
        
        # obs shape(minibatch_size, num_continuous)
        obs = obs.unsqueeze(0).expand(item_emb.shape[0], -1, -1)
        x = torch.cat([item_emb, obs], -1)
        x = self.fc(x)
        prob_ens = self.sigmoid(x).squeeze()
        prob = prob_ens.mean(0)  # ensemble, shape(minibatch_size)
        prob_std = prob_ens.std(0)

        q_itemw = None
        p_itemw = None
        return prob, prob_std, q_item, q_itemw, p_item, p_itemw


class CoverTypeTabCls(nn.Module):

    def __init__(self, model_config, env_config):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        
        self.latent_onehotemb = LatentOneHotEmbedding(model_config, env_config)

        input_dim = self.model_config['num_discrete']*self.model_config['latent_dim'] + self.model_config['num_continuous']
        self.fc = nn.Linear(input_dim, self.model_config['num_class'], bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, cate, cat_incre, obs, cat_nograd=False, sample_size=(10,)):
        q_item, p_item = self.latent_onehotemb.q_E_and_p_E()

        if cat_nograd:
            with torch.no_grad():
                item_emb = self.latent_onehotemb(cate, sample_size=sample_size)
        else:
            item_emb = self.latent_onehotemb(cate, sample_size=sample_size)
        # item_emb  shape(sample_size, minibatch_size, num_discrete, latent_dim)
        
        item_emb_incre = self.latent_onehotemb(cat_incre, sample_size=sample_size)
        item_emb = torch.cat([item_emb, item_emb_incre], -2)

        item_emb = item_emb.reshape(
            *item_emb.shape[:2],
            self.model_config['num_discrete']*self.model_config['latent_dim'])
        
        # obs shape(minibatch_size, num_continuous)
        obs = obs.unsqueeze(0).expand(item_emb.shape[0], -1, -1)
        x = torch.cat([item_emb, obs], -1)
        x = self.fc(x)
        prob_ens = self.softmax(x)
        prob = prob_ens.mean(0)  # ensemble, shape(minibatch_size, )
        prob_std = prob_ens[...,0].std(0)

        q_itemw = None
        p_itemw = None
        return prob, prob_std, q_item, q_itemw, p_item, p_itemw