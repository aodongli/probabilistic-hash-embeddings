import random
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle

from .util import TabDataset, OneHotTabDataset


col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
dtype_dict = {
    'age': np.float32,
    'workclass': 'string',
    'fnlwgt': np.float32,
    'education': 'string',
    'education-num': np.float32,
    'marital-status': 'string',
    'occupation': 'string',
    'relationship': 'string',
    'sex': 'string',
    'capital-gain': np.float32,
    'capital-loss': np.float32,
    'hours-per-week': np.float32,
    'native-country': 'string',
    'label': 'object',
}


class ContinualMushroomData:
    def __init__(self, root, model_config=None, env_config=None):
        self.model_config = model_config
        self.env_config = env_config
        self.root = root
        self.rng = np.random.RandomState(1234)

        df = pd.read_csv(self.root + 'agaricus-lepiota.ori.data', header=0, sep=',', engine='python')
        
        # convert labels
        df['label'] = df['label'].replace({'POISONOUS':1, 'EDIBLE':0})
        
        # # normalize
        # for col in ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']:

        # add prefix to disambiguate same tokens
        cate_cols = ['cap_shape' ,'cap_surface' ,'cap_color' ,'bruises' ,'odor' ,'gill_attachment' ,'gill_spacing' ,'gill_size' ,'gill_color' ,'stalk_shape' ,'stalk_root' ,'stalk_surface_above_ring' ,'stalk_surface_below_ring' ,'stalk_color_above_ring' ,'stalk_color_below_ring' ,'veil_type' ,'veil_color' ,'ring_number' ,'ring_type' ,'spore_print_color' ,'population' ,'habitat']
        for col_name in cate_cols:
            df[col_name] = col_name + '_' + df[col_name].astype(str)

        self.cat_id = {}
        idx = 0
        for col_name in cate_cols:
            cats = df[col_name].unique()
            for cat in cats:
                self.cat_id[cat] = idx
                idx += 1
        print('dictionary size:', idx)
            
        # switch the column's order
        self.columns = ['odor', 'label']  # P_1 rule
        self.df = df[self.columns]
        
        # incremental column:
        self.incre_col = self.env_config.incre_col # 'education' # 'native-country'
        self.incre_col_idx = self.columns.index(self.incre_col) 
        
        # dictionary
        self.dicts = list(self.rng.permutation(df[self.incre_col].unique()))
        

    def get_dataset(self, task_id=0, task_num=1):
        assert task_id < task_num
        
        task_size = np.floor(len(self.dicts)/task_num).astype(np.int32)
        if task_size < 1:
            print('too large task_num, use len(self.dicts)')
            task_size = 1
            task_num = len(self.dicts)
        
        start = task_id*task_size
        if task_id == task_num-1:
            end = len(self.dicts)
        else:
            end = start + task_size
        
        task_cate = self.dicts[start:end]
        print('task categories:', task_cate)
        
        data = self.df[self.df[self.incre_col].isin(task_cate)].to_numpy()
        
        bd = np.ceil(len(data) * 2 / 3).astype(np.int32)
        dat_tr, dat_val, dat_te = data[:bd], data[bd:], data[bd:]
        
        return dat_tr, dat_val, dat_te
    
    
class MushroomDataset(TabDataset):
    def __init__(self, x, y, incre_col_idx, dict_cat=None):
        continuous_col = []  # no continuous values
        discrete_col = list(range(0, 1)) # list(range(0, 22))
        discrete_incre_col = [incre_col_idx]
        discrete_col.remove(incre_col_idx)
        super().__init__(x, y, continuous_col, discrete_col, discrete_incre_col)


class OneHotMushroomDataset(OneHotTabDataset):
    def __init__(self, x, y, incre_col_idx, dict_cat):
        continuous_col = []  # no continuous values
        discrete_col = list(range(0, 1)) # list(range(0, 22))
        discrete_incre_col = [incre_col_idx]
        discrete_col.remove(incre_col_idx)
        super().__init__(x, y, continuous_col, discrete_col, discrete_incre_col, dict_cat)


class EnvConfig:
    incre_col = 'cap_shape'

def main():
    env_config = EnvConfig()
    db = ContinualMushroomData('../data/tabular_data/mushroom/', env_config=env_config)
    df = db.df
    print(df)

    # "job", "marital", "education", "contact", "month", "poutcome"
    discrete_col = ['cap_shape' ,'cap_surface' ,'cap_color' ,'bruises' ,'odor' ,'gill_attachment' ,'gill_spacing' ,'gill_size' ,'gill_color' ,'stalk_shape' ,'stalk_root' ,'stalk_surface_above_ring' ,'stalk_surface_below_ring' ,'stalk_color_above_ring' ,'stalk_color_below_ring' ,'veil_type' ,'veil_color' ,'ring_number' ,'ring_type' ,'spore_print_color' ,'population' ,'habitat']
    dummy_ls = []
    for col in discrete_col:
        dummy_ls.append(pd.get_dummies(df[col]))
    df.drop(discrete_col,
            axis=1, 
            inplace=True)
    df = pd.concat([df, *dummy_ls],
                   axis=1)
    print(df.columns)

    labels = df['label']
    df.drop(['label'],
            axis=1,
            inplace=True)

    df = df.apply(pd.to_numeric)
    labels = labels.apply(pd.to_numeric)

    from sklearn.linear_model import LogisticRegression
    logmodel = LogisticRegression(class_weight='balanced', max_iter=500, random_state=0).fit(df, labels)
    importance = logmodel.coef_.flatten()

    # feature importance
    feat_coef = []
    for n, i in zip(df.columns, importance):
        feat_coef.append([n, i])
        print(n, i)

    feat = pd.DataFrame(feat_coef, columns=['feature', 'coef'])

if __name__ == '__main__':
    main()