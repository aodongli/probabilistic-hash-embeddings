import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle
from datetime import datetime


movie_cat = [
    'Action',
    'Adventure',
    'Animation',
    'Children',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Fantasy',
    'Film-Noir',
    'Horror',
    'IMAX',
    'Musical',
    'Mystery',
    'Romance',
    'Sci-Fi',
    'Thriller',
    'War',
    'Western',
    '(no genres listed)',
]
movie_cat_map = {}
for cid, c in enumerate(movie_cat):
    movie_cat_map[c] = cid

def genre2id(genre_str):
    genres = genre_str.split('|')
    return [movie_cat_map[g] for g in genres]


# type_dict = {  # slow compile

type_dict = {
    'userId': np.int64,
    'movieId': np.int64,
    'rating': np.float64,
    'timestamp': np.int64,
}


class ContinualMovieLens:
    def __init__(self, 
                 root, 
                 model_config=None, 
                 env_config=None):
        self.model_config = model_config
        self.env_config = env_config
        self.root = root
        self.rng = np.random.RandomState(1234)

        df = pd.read_csv(os.path.join(self.root, 'ratings.csv'), header=0, sep=',', engine='pyarrow', dtype=type_dict)  # fast. , date_format='%Y-%m-%d %H:%M:%S'
        self.movie_cat = pd.read_csv(os.path.join(self.root, 'movies.csv'), header=0, sep=',', engine='pyarrow')

        # add movie category id
        self.movie_cat['genres_id'] = self.movie_cat['genres'].apply(genre2id)

        # set date
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # normalize
        for col in ['rating']:
            df[col] = df[col] / df[col].max()

        # add sign to disambiguate same tokens across different columns
        df['movieId'] = -df['movieId']
        self.movie_cat['movieId'] = -self.movie_cat['movieId']

        # get new users and movies arrival time
        self.new_user_arrival = df[['userId', 'timestamp']].groupby('userId').min()
        self.new_movie_arrival = df[['movieId', 'timestamp']].groupby('movieId').min()

        # ready to use
        self.df = df

        # build task info: self.num_task, self.task_map
        self.get_all_task_info()

        # get vocab: self.vocab_map
        self.get_vocab_map()

    
    def get_vocab_map(self):
        cate_cols = ['userId', 'movieId']
        self.vocab_map = {}
        idx = 0
        for col_name in cate_cols:
            cats = self.df[col_name].unique()
            for cat in cats:
                self.vocab_map[cat] = idx
                idx += 1
        print('dictionary size:', idx)


    def get_all_task_info(self):
        self.init_data_start_date = (1996, 1, 1)
        self.init_data_end_date = (2000, 1, 1)

        self.interval = pd.Timedelta(1, unit="d")
        
        self.num_task = 0
        self.task_map = {}
        
        start_date = pd.Timestamp(*self.init_data_end_date)
        end_date = pd.Timestamp(2023, 10, 15)

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


    def _filter_dataframe_by_date(self, 
                                  df, 
                                  start_year, start_month, start_day,
                                  end_year, end_month, end_day,
                                 ):
        def _assert(year, month, day):
            assert year >= 1996 and year <=2023
            assert month >= 1 and month <= 12
            assert day >= 1 and day <= 31
        _assert(start_year, start_month, start_day)
        _assert(end_year, end_month, end_day)
        
        
        start_date = pd.Timestamp(start_year, start_month, start_day)
        end_date = pd.Timestamp(end_year, end_month, end_day)
        
        cond_upper = end_date > df['timestamp']
        cond_lower = df['timestamp'] >= start_date
        filtered_dat = df.loc[cond_lower & cond_upper]
        return filtered_dat


    def get_init_data(self):
        data = self._filter_dataframe_by_date(self.df, 
                                              *self.init_data_start_date,
                                              *self.init_data_end_date,
                                             ).to_numpy()
        tr, val, te = self.train_test_split(data)
        return tr, val, te


    def train_test_split(self, data, te_ratio=0.8):
        # train test split for one task
        perm_idx = self.rng.permutation(np.arange(data.shape[0]))
        data = data[perm_idx]
        
        te_size = int(te_ratio*data.shape[0])
        te = data[:te_size]
        val = data[te_size:]
        tr = val
        return tr, val, te

    
    def get_dataset(self, task_id=0):
        assert task_id < self.num_task

        (start_date, end_date) = self.task_map[task_id]
        data = self._filter_dataframe_by_date(
            self.df,
            start_date.year, start_date.month, start_date.day, 
            end_date.year, end_date.month, end_date.day,
        ).to_numpy()
        
        tr, val, te = self.train_test_split(data)

        return tr, val, te
        
    
class MovieLensDataset(Dataset):
    def __init__(self, x, y, movie_cat, dat_size=None, vocab_map=None):
        del vocab_map

        self.x = x
        self.y = torch.tensor(y.astype(np.float32), dtype=torch.float32)
        self.movie_cat = movie_cat.set_index('movieId')
        
        import hashlib
        self.str_hash = lambda w: int(hashlib.md5(str(w).encode('utf-8')).hexdigest(), 16) % int(1e16)
        self.vec_str_hash = np.vectorize(self.str_hash)

        self.dat_size = dat_size
        self.get_data()

        
    def get_data(self):
        self.cat = []
        cat = self.x[:,:2]
        md5 = self.vec_str_hash(cat) if cat.size else cat.astype(np.int64)
        self.cat = torch.tensor(md5, dtype=torch.int64)


        self.movie_cat_id = self.movie_cat.loc[self.x[:,1]]['genres_id'].to_list()
        def get_multihot(idx_list):
            res = torch.zeros(len(movie_cat))
            res[np.array(idx_list)] = 1.0
            return res
        self.movie_cat_coding = list(map(get_multihot, self.movie_cat_id))

        if not self.dat_size:
            self.dat_size = self.__len__()
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        res = {'cat': self.cat[idx],
               'user_movie': torch.tensor(
                    self.x[idx,:2].astype(np.int64), 
                    dtype=torch.int64),
               'movie_genre': self.movie_cat_coding[idx],  # not of equal size
               'target': self.y[idx],
               'dat_size': self.dat_size,}
        return res


class OneHotMovieLensDataset(Dataset):
    def __init__(self, x, y, movie_cat, dat_size=None, vocab_map=None):
        self.x = x
        self.y = torch.tensor(y.astype(np.float32), dtype=torch.float32)
        self.movie_cat = movie_cat.set_index('movieId')

        self.vocab_map = vocab_map

        self.dat_size = dat_size
        self.get_data()

    def get_cat_id(self, cats):
        res = np.zeros_like(cats, dtype=np.int64)
        for i, cat_list in enumerate(cats):
            for j, cat in enumerate(cat_list):
                res[i,j] = self.vocab_map[cat]
        return res

    def get_data(self):
        self.cat = []
        cat = self.x[:,:2]
        cat_id = self.get_cat_id(cat)
        self.cat = torch.tensor(cat_id, dtype=torch.int64)

        self.movie_cat_id = self.movie_cat.loc[self.x[:,1]]['genres_id'].to_list()
        def get_multihot(idx_list):
            res = torch.zeros(len(movie_cat))
            res[np.array(idx_list)] = 1.0
            return res
        self.movie_cat_coding = list(map(get_multihot, self.movie_cat_id))

        if not self.dat_size:
            self.dat_size = self.__len__()
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        res = {'cat': self.cat[idx],
               'user_movie': torch.tensor(
                    self.x[idx,:2].astype(np.int64), 
                    dtype=torch.int64),
               'movie_genre': self.movie_cat_coding[idx],  # not of equal size
               'target': self.y[idx],
               'dat_size': self.dat_size,}
        return res


def main():
    db = ContinualMovieLens('../data/movielens/')
    print(db.num_task)

    init_dat, val, te = db.get_init_data()
    print(len(init_dat), len(val), len(te))
    print(init_dat)
    tr, _, te = db.get_dataset(0)
    print(len(tr))
    print(len(te))
    print(tr)
    print(te)
    tr, _, te = db.get_dataset(285)
    print(len(tr))
    print(len(te))
    print(tr)
    print(te)

    dataset = MovieLensDataset(te, te[:,2], db.movie_cat)
    print(dataset[0])


if __name__ == '__main__':
    main()