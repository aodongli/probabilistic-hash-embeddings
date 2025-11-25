import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.poisson import Poisson
from torch.distributions.geometric import Geometric
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

class NELBO_LatentBinCls(nn.Module):
    def __init__(self, model_config=None, env_config=None):
        super(NELBO_LatentBinCls, self).__init__()
        self.reg_weight = env_config.reg_weight

    def forward(self, dat_size, prob, label, q_cond, q_weight, p_cond, p_weight, verbose=False):
        ll_tot = -F.binary_cross_entropy(prob, label, reduction='none')  # shape(minibatch_size)
        kl_cond = kl_divergence(q_cond, p_cond).sum()
        if q_weight is None or p_weight is None:
            kl = kl_cond
        else:
            kl_weight = kl_divergence(q_weight, p_weight).sum()
            kl = kl_cond + kl_weight
        ll = torch.mean(ll_tot*dat_size)
        elbo = ll - self.reg_weight*kl
        nelbo = -elbo
        nll = -ll

        if verbose:
            return nelbo, nll, kl, ll_tot
        else:
            return nelbo, nll, kl


class NELBO_LatentMultiCls(nn.Module):
    def __init__(self, model_config=None, env_config=None):
        super(NELBO_LatentMultiCls, self).__init__()
        self.reg_weight = env_config.reg_weight

    def forward(self, dat_size, prob, label, q_cond, q_weight, p_cond, p_weight, verbose=False):
        label = label.type(torch.LongTensor)
        idx = torch.arange(label.shape[0]).to(label)
        ll_tot = torch.log(prob[idx, label]+1e-8)  # shape(minibatch_size)
        kl_cond = kl_divergence(q_cond, p_cond).sum()
        if q_weight is None or p_weight is None:
            kl = kl_cond
        else:
            kl_weight = kl_divergence(q_weight, p_weight).sum()
            kl = kl_cond + kl_weight
        ll = torch.mean(ll_tot*dat_size)
        elbo = ll - self.reg_weight*kl
        nelbo = -elbo
        nll = -ll

        if verbose:
            return nelbo, nll, kl, ll_tot
        else:
            return nelbo, nll, kl


class NELBO_LatentTimeFilter(nn.Module):
    def __init__(self, model_config=None, env_config=None):
        super(NELBO_LatentTimeFilter, self).__init__()
        self.reg_weight = env_config.reg_weight

    def forward(self, dat_size, rate, target, q_cond, q_weight, q_time, p_cond, p_weight, p_time, verbose=False):
        dat_size = dat_size[0]

        poisson_dist = Poisson(rate)
        ll_tot = poisson_dist.log_prob(target) 
        kl_time = kl_divergence(q_time, p_time).sum(-1)
        kl_time = torch.mean(kl_time)*dat_size  # reweight
        kl_cond = kl_divergence(q_cond, p_cond).sum()
        if q_weight is None or p_weight is None:
            kl_hash = kl_cond
        else:
            kl_weight = kl_divergence(q_weight, p_weight).sum()
            kl_hash = kl_cond + kl_weight
        kl = self.reg_weight*kl_hash + kl_time
        ll = torch.mean(ll_tot*dat_size)  # reweight
        elbo = ll - kl
        nelbo = -elbo
        nll = -ll

        if verbose:
            return nelbo, nll, kl, ll_tot
        else:
            return nelbo, nll, kl


class NELBO_LatentRegression(nn.Module):
    def __init__(self, model_config=None, env_config=None):
        super(NELBO_LatentRegression, self).__init__()
        self.device = torch.device(model_config['device'])
        self.reg_weight = env_config.reg_weight
        self.ll_scale = torch.tensor([model_config['ll_scale']]).to(self.device)

    def forward(self, dat_size, loc, target, q_cond, q_weight, p_cond, p_weight, verbose=False):
        dat_size = dat_size[0]

        gaussian_dist = Normal(loc, scale=self.ll_scale)  # hyperparamter
        ll_tot = gaussian_dist.log_prob(target)  # ensemble with lowerbound mixture dist
        ll_tot = ll_tot.mean(0)  # shape: (mini_batch,)
        kl_cond = kl_divergence(q_cond, p_cond).sum()
        if q_weight is None or p_weight is None:
            kl_hash = kl_cond
        else:
            kl_weight = kl_divergence(q_weight, p_weight).sum()
            kl_hash = kl_cond + kl_weight
        kl = self.reg_weight*kl_hash
        ll = torch.mean(ll_tot*dat_size)  # reweight
        
        elbo = ll - kl
        nelbo = -elbo
        nll = -ll
        mae = torch.abs(loc - target).mean()

        if verbose:
            return nelbo, nll, kl, mae, ll_tot
        else:
            return nelbo, nll, kl, mae