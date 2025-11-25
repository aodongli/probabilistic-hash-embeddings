import random
import copy
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle
from datetime import datetime



type_dict = {
    'InvoiceNo': 'string',
    'StockCode': 'string',
    'Description': 'string',
    'Quantity': np.int32,
    'UnitPrice': np.float32,
    'CustomerID': 'string',
    'Country': 'string',
}


class OnlineRetailData:
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
        self.dicts = list(df[self.incre_col].unique())
        
        # seq_len
        self.seq_len = self.model_config['sample_seq_len']

        self.get_all_task_info()
        self.seq_data = self.construct_time_series_data()
        print('Finish constructing time series data of shape', self.seq_data.shape)



    def get_all_task_info(self):
        self.init_data_start_date = (2010, 12, 1)
        self.init_data_end_date = (2011, 2, 1)

        self.interval = pd.Timedelta(1, unit="d")
        
        self.num_task = 0
        self.task_map = {}
        
        start_date = pd.Timestamp(*self.init_data_end_date)
        end_date = pd.Timestamp(2011, 12, 10)

        task_id = 0
        cur = start_date
        while cur < end_date:
            next = cur + self.interval
            # build map
            self.task_map[task_id] = [cur, next]
            self.num_task += 1

            # increase month
            cur = next
            task_id += 1


    def _expand_array(self, a, target_len):
        to_fill = np.empty((target_len, a.shape[1]), dtype=object)

        to_fill[:target_len-len(a)] = np.repeat(a[[0]], target_len-len(a), axis=0)
        to_fill[target_len-len(a):] = a
        return to_fill


    def construct_time_series_data(self):
        # construct time series for each item
        dataset = []
        for task in self.dicts:
            task_df = self.df[self.df[self.incre_col].isin([task])].to_numpy()
            
            if len(task_df) < self.seq_len:
                dataset.append(
                        self._expand_array(task_df, self.seq_len)
                )
                continue

            for i in range(len(task_df) - self.seq_len + 1):
                dataset.append(
                        task_df[i:i+self.seq_len]
                )

        # sort against the end time
        dataset.sort(key=lambda x: x[-1][0])

        return np.asarray(dataset)

    
    def _filter_dataframe_by_date(self, 
                                  df, 
                                  start_year, start_month, start_day,
                                  end_year, end_month, end_day,
                                 ):
        def _assert(year, month, day):
            assert year >= 2010 and year <=2021
            assert month >= 1 and month <= 12
            assert day >= 1 and day <= 31
        _assert(start_year, start_month, start_day)
        _assert(end_year, end_month, end_day)
        
        start_date = np.datetime64(datetime(start_year, start_month, start_day))
        end_date = np.datetime64(datetime(end_year, end_month, end_day))
        
        cond_upper = end_date > df[:,-1,0]
        cond_lower = df[:,-1,0] >= start_date
        filtered_dat = df[cond_lower & cond_upper]
        return filtered_dat


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
        data = self._filter_dataframe_by_date(self.seq_data, 
                                              *self.init_data_start_date,
                                              *self.init_data_end_date,
                                             )
        tr, val, te = self.train_test_split(data)
        return tr, val, te


    def get_dataset(self, task_id=0):
        assert task_id < self.num_task

        (start_date, end_date) = self.task_map[task_id]
        data = self._filter_dataframe_by_date(
            self.seq_data,
            start_date.year, start_date.month, start_date.day, 
            end_date.year, end_date.month, end_date.day,
        )
        
        tr, val, te = self.train_test_split(data)

        return tr, val, te
    
    
class OnlineRetailDataset:
    def __init__(self, x, y, incre_col_idx, dict_cat=None):
        self.x = x
        self.y = torch.tensor(y.astype(np.float32), dtype=torch.float32)
        
        self.time_col_list = [0]
        self.continuous_col = [1]
        self.discrete_col = [2,3,4]
        self.discrete_incre_col = [incre_col_idx]
        self.discrete_col.remove(incre_col_idx)

        import hashlib
        self.str_hash = lambda w: int(hashlib.md5(str(w).encode('utf-8')).hexdigest(), 16) % int(1e16)
        self.vec_str_hash = np.vectorize(self.str_hash)

    def time_normalize(self, t):
        mins = t.astype('timedelta64[m]')
        mins = mins / np.timedelta64(1, 'm')
        w = torch.tensor(mins/60./24/7)
        w = torch.clamp(w, min=0., max=3.)
        return w  # one day as unit: 86400=24*60*60
                        # one hour as unit: 3600=60*60
                        # one minute as unit: 60

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        target = self.y[idx]
        obs = torch.tensor(x[:, self.continuous_col].astype(np.float32))
        cat = x[:, self.discrete_col]
        md5 = self.vec_str_hash(cat) if cat.size else cat.astype(np.int64)
        cat = torch.tensor(md5, dtype=torch.int64)
        cat_incre = x[:, self.discrete_incre_col]
        md5 = self.vec_str_hash(cat_incre)
        cat_incre = torch.tensor(md5, dtype=torch.int64)
        time = self.time_normalize(
            x[:,0] - np.concatenate([x[[0],0], x[:-1,0]])
        ).unsqueeze(-1)
        res = {'obs': obs,
               'cat': cat,
               'cat_incre': cat_incre,
               'target': target,
               'time': time,
               'dat_size': self.__len__(),}
        return res



class OnlineOneHotRetailDataset:
    def __init__(self, x, y, incre_col_idx, dict_cat):
        self.x = x
        self.y = torch.tensor(y.astype(np.float32), dtype=torch.float32)
        self.dict_cat = dict_cat
        
        self.time_col_list = [0]
        self.continuous_col = [1]
        self.discrete_col = [2,3,4]
        self.discrete_incre_col = [incre_col_idx]
        self.discrete_col.remove(incre_col_idx)

    def get_cat_id(self, cats):
        res = np.zeros_like(cats, dtype=np.int64)
        for i, cat_list in enumerate(cats):
            for j, cat in enumerate(cat_list):
                res[i,j] = self.dict_cat[cat]
        return res

    def time_normalize(self, t):
        mins = t.astype('timedelta64[m]')
        mins = mins / np.timedelta64(1, 'm')
        w = torch.tensor(mins/60./24/7)
        w = torch.clamp(w, min=0., max=3.)
        return w  # one day as unit: 86400=24*60*60
                        # one hour as unit: 3600=60*60
                        # one minute as unit: 60

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        target = self.y[idx]
        obs = torch.tensor(x[:, self.continuous_col].astype(np.float32))
        cat = x[:, self.discrete_col]
        md5 = self.get_cat_id(cat)
        cat = torch.tensor(md5, dtype=torch.int64)
        cat_incre = x[:, self.discrete_incre_col]
        md5 = self.get_cat_id(cat_incre)
        cat_incre = torch.tensor(md5, dtype=torch.int64)
        time = self.time_normalize(
            x[:,0] - np.concatenate([x[[0],0], x[:-1,0]])
        ).unsqueeze(-1)
        res = {'obs': obs,
               'cat': cat,
               'cat_incre': cat_incre,
               'target': target,
               'time': time,
               'dat_size': self.__len__(),}
        return res


class EnvConfig:
    incre_col = 'StockCode'

def main():
    env_config = EnvConfig()
    db = OnlineRetailData('../data/time_tabular_data/retail/', env_config=env_config, model_config={'sample_seq_len': 10})
    print(db.num_task)
    
    tr, _, te = db.get_init_data()
    print(len(tr), len(te))

    tr, _, te = db.get_dataset(0)
    print(len(tr), len(te))

    dataset = OnlineRetailDataset(te[..., :-1], te[..., -1], db.incre_col_idx)
    print(dataset[0])

    dataset = OnlineOneHotRetailDataset(te[..., :-1], te[..., -1], db.incre_col_idx, db.cat_id)
    print(dataset[0])



if __name__ == '__main__':
    main()