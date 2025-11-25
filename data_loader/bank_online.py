import random
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle


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


class OnlineBankData:
    def __init__(self, root, model_config=None, env_config=None):
        self.model_config = model_config
        self.env_config = env_config
        self.root = root
        self.rng = np.random.RandomState(1234)

        df = pd.read_csv(self.root + 'bank-full.csv', header=0, sep=';', engine='python')
        
        # convert labels
        df['y'] = df['y'].replace({'no':0, 'yes':1})
        
        # normalize
        for col in ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']:
            df[col] = df[col] / df[col].max()

        # add prefix to disambiguate same tokens
        cate_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
        for col_name in cate_cols:
            df[col_name] = col_name + '_' + df[col_name].astype(str)

        # summarize dictionary
        self.cat_id = {}
        idx = 0
        for col_name in cate_cols:
            cats = df[col_name].unique()
            for cat in cats:
                self.cat_id[cat] = idx
                idx += 1
        print('dictionary size:', idx)
            
        # switch the column's order
        self.columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', "job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome",  'y']
        self.df = df[self.columns]
        
        # incremental column:
        self.incre_col = self.env_config.incre_col # 'education' # 'native-country'
        self.incre_col_idx = self.columns.index(self.incre_col) 
        
        # incremental dictionary
        self.dicts = list(self.rng.permutation(df[self.incre_col].unique()))
        self.get_all_task_info()
        print('Number of tasks:', self.num_task)

        # # add noise
        # if self.env_config.noise_obs:
        

    def add_noise(self, data, ratio=0.3):
        num_flip = int(len(data) * ratio)
        flip_idx = self.rng.permutation(len(data))[:num_flip]
        label = np.array(data[:,-1], dtype=np.float32)
        label[flip_idx] = 1 - label[flip_idx]
        data[:,-1] = label
        return data

    def get_all_task_info(self):
        self.data_size = self.df.shape[0]

        self.init_task_ratio = 0.3
        self.init_task_size = int(self.data_size*self.init_task_ratio)

        self.subsequent_task_size = 100
        self.num_task = np.ceil(
            (self.data_size - self.init_task_size) / self.subsequent_task_size
        ).astype(np.int32)

        # random permutation
        self.data = self.df.to_numpy()
        perm_idx = self.rng.permutation(np.arange(self.data.shape[0]))
        self.data = self.data[perm_idx]

    def train_test_split(self, data, te_ratio=0.8):
        # train test split for one task
        perm_idx = self.rng.permutation(np.arange(data.shape[0]))
        data = data[perm_idx]
        
        te_size = int(te_ratio*data.shape[0])
        te = data[:te_size]
        val = data[te_size:]
        tr = val
        return tr, val, te

    def get_init_data(self):
        data = self.data[:self.init_task_size]
        tr, val, te = self.train_test_split(data)
        return tr, val, te

    def get_dataset(self, task_id=0):
        assert task_id < self.num_task
        
        sub_data = self.data[self.init_task_size:]
        
        start = task_id*self.subsequent_task_size
        end = (task_id+1)*self.subsequent_task_size
        
        data = sub_data[start:end]
        if self.env_config.random_task_size:
            resize = self.rng.randint(5, self.subsequent_task_size+1)
            data = data[:resize]
        if self.env_config.noise_obs:
            if self.rng.uniform() < 0.2:
                data = self.add_noise(data, ratio=0.3)
        tr, val, te = self.train_test_split(data)
        return tr, val, te
    

class EnvConfig:
    incre_col = 'poutcome'

def main():
    env_config = EnvConfig()
    db = OnlineBankData('../data/tabular_data/bank/', env_config=env_config)
    df = db.df
    print(df)


if __name__ == '__main__':
    main()