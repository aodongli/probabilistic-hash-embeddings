import numpy as np
import torch 
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'data': self.x[idx],
                'label': self.y[idx]}


class TabDataset(Dataset):
    def __init__(self, x, y, 
                 continuous_col_list, discrete_col_list, discrete_incre_col_list):
        self.x = x
        self.y = torch.tensor(y.astype(np.float32), dtype=torch.float32)
        
        self.continuous_col = continuous_col_list
        self.discrete_col = discrete_col_list
        self.discrete_incre_col = discrete_incre_col_list
        
        import hashlib
        self.str_hash = lambda w: int(hashlib.md5(str(w).encode('utf-8')).hexdigest(), 16) % int(1e16)
        self.vec_str_hash = np.vectorize(self.str_hash)
        
        self.get_data()
        
    def get_data(self):
        self.cat = []
        self.cat_incre = []
        self.obs = []
        
        cat = self.x[:, self.discrete_col]
        md5 = self.vec_str_hash(cat) if cat.size else cat.astype(np.int64)
        self.cat = torch.tensor(md5, dtype=torch.int64)
        cat_incre = self.x[:, self.discrete_incre_col]
        md5 = self.vec_str_hash(cat_incre)
        self.cat_incre = torch.tensor(md5, dtype=torch.int64)
        obs = self.x[:, self.continuous_col]
        self.obs = torch.tensor(obs.astype(np.float32), dtype=torch.float32)

        

    def __len__(self):
        return len(self.obs)
        
    def __getitem__(self, idx):
        res = {'obs': self.obs[idx],
               'cat': self.cat[idx],
               'cat_incre': self.cat_incre[idx],
               'label': self.y[idx],
               'dat_size': self.__len__(),}
        return res


class OneHotTabDataset(Dataset):
    def __init__(self, x, y, 
                 continuous_col_list, discrete_col_list, discrete_incre_col_list, 
                 dict_cat):
        self.x = x
        self.y = torch.tensor(y.astype(np.float32), dtype=torch.float32)
        
        self.continuous_col = continuous_col_list
        self.discrete_col = discrete_col_list
        self.discrete_incre_col = discrete_incre_col_list
        self.dict_cat = dict_cat
        
        self.get_data()

    def get_cat_id(self, cats):
        res = np.zeros_like(cats, dtype=np.int64)
        for i, cat_list in enumerate(cats):
            for j, cat in enumerate(cat_list):
                res[i,j] = self.dict_cat[cat]
        return res
        
    def get_data(self):
        self.cat = []
        self.cat_incre = []
        self.obs = []
        
        cat = self.x[:, self.discrete_col]
        cat_idx = self.get_cat_id(cat)
        self.cat = torch.tensor(cat_idx, dtype=torch.int64)
        cat_incre = self.x[:, self.discrete_incre_col]
        cat_idx = self.get_cat_id(cat_incre)
        self.cat_incre = torch.tensor(cat_idx, dtype=torch.int64)
        obs = self.x[:, self.continuous_col]
        self.obs = torch.tensor(obs.astype(np.float32), dtype=torch.float32)

        

    def __len__(self):
        return len(self.obs)
        
    def __getitem__(self, idx):
        res = {'obs': self.obs[idx],
               'cat': self.cat[idx],
               'cat_incre': self.cat_incre[idx],
               'label': self.y[idx],
               'dat_size': self.__len__(),}
        return res



class TimeTabDataset(Dataset):
    def __init__(self, x, y,
                 time_col_list, continuous_col_list, 
                 discrete_col_list, discrete_incre_col_list,
                 seq_len=10):
        self.x = x
        self.y = torch.tensor(y.astype(np.float32), dtype=torch.float32)
        self.seq_len = seq_len

        self.time_col = time_col_list
        self.continuous_col = continuous_col_list
        self.discrete_col = discrete_col_list
        self.discrete_incre_col = discrete_incre_col_list
        
        import hashlib
        self.str_hash = lambda w: int(hashlib.md5(str(w).encode('utf-8')).hexdigest(), 16) % int(1e16)
        self.vec_str_hash = np.vectorize(self.str_hash)
        
        self.get_data()
        
    def get_data(self):
        self.time = []
        self.cat = []
        self.cat_incre = []
        self.obs = []
        
        self.time = self.x[:, self.time_col]
        cat = self.x[:, self.discrete_col]
        md5 = self.vec_str_hash(cat) if cat.size else cat.astype(np.int64)
        self.cat = torch.tensor(md5, dtype=torch.int64)
        cat_incre = self.x[:, self.discrete_incre_col]
        md5 = self.vec_str_hash(cat_incre)
        self.cat_incre = torch.tensor(md5, dtype=torch.int64)
        obs = self.x[:, self.continuous_col]
        self.obs = torch.tensor(obs.astype(np.float32), dtype=torch.float32)

        

    def __len__(self):
        return len(self.obs) - self.seq_len + 1

    def time_normalize(self, t):
        mins = t.astype('timedelta64[m]')
        mins = mins / np.timedelta64(1, 'm')
        w = torch.tensor(mins/60./24/7)
        w = torch.clamp(w, min=0., max=3.)
        return w  # one day as unit: 86400=24*60*60
                        # one hour as unit: 3600=60*60
                        # one minute as unit: 60
        
    def __getitem__(self, idx):
        # sample a trajectory of size L
        obs = [self.obs[idx]]
        cat = [self.cat[idx]]
        cat_incre = [self.cat_incre[idx]]
        time = [self.time_normalize(self.time[idx]-self.time[idx])]
        target = [self.y[idx]]
        for i in range(idx+1, idx+self.seq_len):
            obs.append(self.obs[i])
            cat.append(self.cat[i])
            cat_incre.append(self.cat_incre[i])
            time.append(self.time_normalize(self.time[i]-self.time[i-1]))
            target.append(self.y[i])

        res = {'obs': torch.stack(obs, 0),
               'cat': torch.stack(cat, 0),
               'cat_incre': torch.stack(cat_incre, 0),
               'target': torch.stack(target, 0),
               'time': torch.stack(time, 0),
               'dat_size': self.__len__(),}
        return res



class OneHotTimeTabDataset(Dataset):
    def __init__(self, x, y,
                 time_col_list, continuous_col_list, 
                 discrete_col_list, discrete_incre_col_list,
                 dict_cat,
                 seq_len=10):
        self.x = x
        self.y = torch.tensor(y.astype(np.float32), dtype=torch.float32)
        self.seq_len = seq_len
        self.dict_cat = dict_cat

        self.time_col = time_col_list
        self.continuous_col = continuous_col_list
        self.discrete_col = discrete_col_list
        self.discrete_incre_col = discrete_incre_col_list
        
        self.get_data()

    def get_cat_id(self, cats):
        res = np.zeros_like(cats, dtype=np.int64)
        for i, cat_list in enumerate(cats):
            for j, cat in enumerate(cat_list):
                res[i,j] = self.dict_cat[cat]
        return res
        
    def get_data(self):
        self.time = []
        self.cat = []
        self.cat_incre = []
        self.obs = []
        
        self.time = self.x[:, self.time_col]
        cat = self.x[:, self.discrete_col]
        md5 = self.get_cat_id(cat)
        self.cat = torch.tensor(md5, dtype=torch.int64)
        cat_incre = self.x[:, self.discrete_incre_col]
        md5 = self.get_cat_id(cat_incre)
        self.cat_incre = torch.tensor(md5, dtype=torch.int64)
        obs = self.x[:, self.continuous_col]
        self.obs = torch.tensor(obs.astype(np.float32), dtype=torch.float32)

        

    def __len__(self):
        return len(self.obs) - self.seq_len + 1

    def time_normalize(self, t):
        mins = t.astype('timedelta64[m]')
        mins = mins / np.timedelta64(1, 'm')
        w = torch.tensor(mins/60./24/7)
        w = torch.clamp(w, min=0., max=3.)
        return w  # one day as unit: 86400=24*60*60
                        # one hour as unit: 3600=60*60
                        # one minute as unit: 60
        
    def __getitem__(self, idx):
        # sample a trajectory of size L
        obs = [self.obs[idx]]
        cat = [self.cat[idx]]
        cat_incre = [self.cat_incre[idx]]
        time = [self.time_normalize(self.time[idx]-self.time[idx])]
        target = [self.y[idx]]
        for i in range(idx+1, idx+self.seq_len):
            obs.append(self.obs[i])
            cat.append(self.cat[i])
            cat_incre.append(self.cat_incre[i])
            time.append(self.time_normalize(self.time[i]-self.time[i-1]))
            target.append(self.y[i])

        res = {'obs': torch.stack(obs, 0),
               'cat': torch.stack(cat, 0),
               'cat_incre': torch.stack(cat_incre, 0),
               'target': torch.stack(target, 0),
               'time': torch.stack(time, 0),
               'dat_size': self.__len__(),}
        return res