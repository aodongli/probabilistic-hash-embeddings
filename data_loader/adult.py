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


class ContinualAdultData:
    def __init__(self, root, model_config=None, env_config=None):
        self.model_config = model_config
        self.env_config = env_config
        self.root = root
        self.rng = np.random.RandomState(1234)
        
        df_tr = pd.read_csv(self.root + 'adult.data', names=col_names, na_values=['?'], sep=', ', dtype=dtype_dict, engine='python')
        df_te = pd.read_csv(self.root + 'adult.test', names=col_names, na_values=['?'], sep=', ', dtype=dtype_dict, engine='python')
        df = pd.concat([df_tr, df_te])
        
        # remove missing values 
        df = df.dropna()
        
        # convert labels
        df['label'] = df['label'].replace({'<=50K':0, '>50K':1})
        
        # normalize
        for col in ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
            df[col] = df[col] / df[col].max()

        # add prefix to disambiguate same tokens
        cate_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
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
        self.columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'label']
        self.df = df[self.columns]
        
        # incremental column:
        self.incre_col = self.env_config.incre_col # 'education' # 'native-country' 
        self.incre_col_idx = self.columns.index(self.incre_col)
        
        # dictionary
        self.dicts = list(self.rng.permutation(df[self.incre_col].unique()))
        # self.dicts = ['Preschool', 
        #               '7th-8th', 
        #               '9th', 
        #               '11th', 
        #               '10th', 
        #               '1st-4th', 
        #               '12th', 
        #               '5th-6th', 
        #               'HS-grad', 
        #               'Assoc-acdm', 
        #               'Assoc-voc', 
        #               'Some-college', 
        #               'Bachelors', 
        #               'Masters', 
        #               'Doctorate', 
        #               'Prof-school']  # sort against feature importance
        # self.dicts = ['education_Bachelors', 
        #               'education_Masters', 
        #               'education_Doctorate', 
        #               'education_Preschool', 
        #               'education_1st-4th', 
        #               'education_5th-6th', 
        #               'education_7th-8th', 
        #               'education_9th', 
        #               'education_10th', 
        #               'education_11th', 
        #               'education_12th', 
        #               'education_HS-grad', 
        #               'education_Prof-school',
        #               'education_Assoc-acdm', 
        #               'education_Assoc-voc', 
        #               'education_Some-college', 
        #               ]  # complete list
        # self.dicts = ['education_Bachelors',
        #             #   'education_Masters', 
        #             #   'education_Doctorate', 
        #               'education_Preschool', 
        #             #   'education_1st-4th', 
        #             #   'education_5th-6th', 
        #             #   'education_7th-8th', 
        #               'education_9th', 
        #             #   'education_10th', 
        #             #   'education_11th', 
        #             #   'education_12th', 
        #               'education_HS-grad', 
        #               'education_Prof-school',
        #             #   'education_Assoc-acdm', 
        #             #   'education_Assoc-voc', 
        #             #   'education_Some-college', 
        #               ]    # a more obvious pattern for motivating the problem
        

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
    
    
class AdultDataset(TabDataset):
    def __init__(self, x, y, incre_col_idx, dict_cat=None):
        continuous_col = list(range(0, 6))
        discrete_col = list(range(6, 14))
        discrete_incre_col = [incre_col_idx]
        discrete_col.remove(incre_col_idx)
        super().__init__(x, y, continuous_col, discrete_col, discrete_incre_col)

class OneHotAdultDataset(OneHotTabDataset):
    def __init__(self, x, y, incre_col_idx, dict_cat):
        continuous_col = list(range(0, 6))
        discrete_col = list(range(6, 14))
        discrete_incre_col = [incre_col_idx]
        discrete_col.remove(incre_col_idx)
        super().__init__(x, y, continuous_col, discrete_col, discrete_incre_col, dict_cat)


class EnvConfig:
    incre_col = 'education'

def main():
    env_config = EnvConfig()
    db = ContinualAdultData('../data/tabular_data/adult/', env_config=env_config)
    df = db.df

    # 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 
    workclass = pd.get_dummies(df['workclass'], prefix='workclass')
    education = pd.get_dummies(df['education'], prefix='education')
    marital_status = pd.get_dummies(df['marital-status'], prefix='marital')
    occupation = pd.get_dummies(df['occupation'], prefix='occupation')
    relationship = pd.get_dummies(df['relationship'], prefix='relationship')
    race = pd.get_dummies(df['race'], prefix='race')
    sex = pd.get_dummies(df['sex'], prefix='sex')
    native_country = pd.get_dummies(df['native-country'], prefix='country')
    df.drop(['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'],
            axis=1, 
            inplace=True)
    df = pd.concat([df, workclass, education, marital_status, occupation, relationship, race, sex, native_country],
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
    print(feat[feat['feature'].str.contains('education_')].sort_values(['coef']))

if __name__ == '__main__':
    main()