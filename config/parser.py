# register your model in this script
# import the model class
from models.tabular_cls import AdultTabBinCls, CoverTypeTabCls
from models.tabular_cls_onehot import AdultTabBinCls as OneHotAdultTabBinCls
from models.tabular_cls_onehot import CoverTypeTabCls as OneHotCoverTypeTabCls
from models.time_tab_pred import RetailTabPred
from models.time_tab_pred_onehot import RetailTabPred as OneHotRetailTabPred
from models.movielens_ncf import MovieLensNCF_MLP, MovieLensNCF_MLP2, MovieLensNCF_MLP2_AugInput
from models.movielens_ncf_onehot import MovieLensNCF_MLP2_AugInput as OneHot_MovieLensNCF_MLP2_AugInput
from trainers.tab_cls_trainer import DyTabBinCls_Trainer
from trainers.time_tab_pred_trainer import TimeDyTabPred_Trainer
from trainers.online_tab_pred_trainer import OnlineDyTabPred_Trainer
from trainers.online_time_tab_pred_trainer import OnlineTimeDyTabPred_Trainer
from trainers.movielens_recommend_trainer import MovieLensRecommendTrainer
from losses.vi_loss import NELBO_LatentBinCls, NELBO_LatentMultiCls, NELBO_LatentTimeFilter
from losses.vi_loss import NELBO_LatentRegression

model_map = {
    'adult_cls': AdultTabBinCls,
    'covertype_cls': CoverTypeTabCls,
    'retail_pred': RetailTabPred,
    'onehot_adult_cls': OneHotAdultTabBinCls,
    'onehot_covertype_cls': OneHotCoverTypeTabCls,
    'onehot_retail_pred': OneHotRetailTabPred,
    'movielens_ncf_mlp': MovieLensNCF_MLP,
    'movielens_ncf_mlp2': MovieLensNCF_MLP2,
    'movielens_ncf_mlp2_auginput': MovieLensNCF_MLP2_AugInput,
    'movielens_ncf_mlp2_auginput_onehot': OneHot_MovieLensNCF_MLP2_AugInput,
}

trainer_map = {
    'dynamic_tabular_trainer': DyTabBinCls_Trainer,
    'time_dytab_trainer': TimeDyTabPred_Trainer,
    'online_dytab_trainer': OnlineDyTabPred_Trainer,
    'online_time_dytab_trainer': OnlineTimeDyTabPred_Trainer,
    'movielens_recommend_trainer': MovieLensRecommendTrainer,
}

loss_map = {
    'nelbo_bce': NELBO_LatentBinCls,
    'nelbo_ce': NELBO_LatentMultiCls,
    'nelbo_time': NELBO_LatentTimeFilter,
    'nelbo_regression': NELBO_LatentRegression,
}


from torch.optim import Adam, SGD


optim_map = {
    'adam': Adam,
    'sgd': SGD,
}


import os
from pathlib import Path
import json
import yaml
import pickle
import numpy as np

def model_config_reader(config_file_name):
    # return a dict configuration
    model_config = None
    if isinstance(config_file_name, dict):
        model_config =  config_file_name

    path = Path(os.path.join('config_files', config_file_name))
    if path.suffix == ".json":
        model_config =  json.load(open(path, "r"))
    elif path.suffix in [".yaml", ".yml"]:
        model_config =  yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    elif path.suffix in [".pkl", ".pickle"]:
        model_config =  pickle.load(open(path, "rb"))
    else:
        raise ValueError("Only JSON, YaML and pickle files supported.")

    model_config['model_class'] = model_map[model_config['model']]
    model_config['trainer_class'] = trainer_map[model_config['trainer']]
    model_config['loss_class'] = loss_map[model_config['loss']]
    model_config['optim_class'] = optim_map[model_config['optim']]

    return model_config