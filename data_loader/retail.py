import random
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

from .util import TimeTabDataset, OneHotTimeTabDataset


type_dict = {
    'InvoiceNo': 'string',
    'StockCode': 'string',
    'Description': 'string',
    'Quantity': np.int32,
    'UnitPrice': np.float32,
    'CustomerID': 'string',
    'Country': 'string',
}


class ContinualRetailData:
    def __init__(self, root, model_config=None, env_config=None):
        self.model_config = model_config
        self.env_config = env_config
        self.root = root
        self.rng = np.random.RandomState(1234)

        df = pd.read_csv(self.root + 'Online_Retail.csv', header=0, sep=',', engine='python', parse_dates=['InvoiceDate'], dtype=type_dict)  # fast. , date_format='%Y-%m-%d %H:%M:%S'
        
        # sort by goods items
        df = df.sort_values(by=['StockCode', 'InvoiceDate'])

        # filter
        df = df[df['Quantity'] > 0]
        
        # normalize
        for col in ['UnitPrice']:
            df[col] = df[col] / df[col].max()

        # add prefix to disambiguate same tokens
        cate_cols = ['StockCode', 'CustomerID', 'Country']
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
        self.columns = ['InvoiceDate', 'UnitPrice', 'StockCode', 'CustomerID', 'Country', 'Quantity']
        self.df = df[self.columns]
        
        # incremental column:
        self.incre_col = self.env_config.incre_col # 'education' # 'native-country'
        self.incre_col_idx = self.columns.index(self.incre_col) 
        
        # dictionary
        self.dicts = list(self.rng.permutation(df[self.incre_col].unique()))
        

    def train_test_split(self, task_cate):
        # task-wise train test split
        dat_tr, dat_val, dat_te = [], [], []
        for task in task_cate:
            task_df = self.df[self.df[self.incre_col].isin([task])].to_numpy()
            bd = np.ceil(len(task_df) * 2 / 3).astype(np.int32)
            task_tr, task_val, task_te = task_df[:bd], task_df[bd:], task_df[bd:]

            dat_tr.append(task_tr)
            dat_te.append(task_te)
            dat_val.append(task_val)

        dat_tr = np.concatenate(dat_tr)
        dat_te = np.concatenate(dat_te)
        dat_val = np.concatenate(dat_val)
        return dat_tr, dat_val, dat_te

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
        print('task categories (top 10):', task_cate[:10])

        dat_tr, dat_val, dat_te = self.train_test_split(task_cate)

        
        
        
        return dat_tr, dat_val, dat_te
    
    
class RetailDataset(TimeTabDataset):
    def __init__(self, x, y, incre_col_idx, dict_cat=None, seq_len=10):
        time_col_list = [0]
        continuous_col = [1]
        discrete_col = [2,3,4]
        discrete_incre_col = [incre_col_idx]
        discrete_col.remove(incre_col_idx)
        super().__init__(x, y, 
                         time_col_list, continuous_col, discrete_col, discrete_incre_col,
                         seq_len=seq_len)


class OneHotRetailDataset(OneHotTimeTabDataset):
    def __init__(self, x, y, incre_col_idx, dict_cat, seq_len=10):
        time_col_list = [0]
        continuous_col = [1]
        discrete_col = [2,3,4]
        discrete_incre_col = [incre_col_idx]
        discrete_col.remove(incre_col_idx)
        super().__init__(x, y, 
                         time_col_list, continuous_col, discrete_col, discrete_incre_col,
                         dict_cat,
                         seq_len=seq_len)


class EnvConfig:
    incre_col = 'StockCode'

def main():
    env_config = EnvConfig()
    db = ContinualRetailData('../data/time_tabular_data/retail/', env_config=env_config)
    
    tr, _, te = db.get_dataset(0, 10)
    print(tr)
    print(te)

    dataset = RetailDataset(te[:, :-1], te[:,-1], db.incre_col_idx)
    print(dataset[0])


if __name__ == '__main__':
    main()