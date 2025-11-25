import numpy as np
import torch 
from .adult import ContinualAdultData, AdultDataset, OneHotAdultDataset
from .adult_online import OnlineAdultData
from .bank import ContinualBankData, BankDataset, OneHotBankDataset
from .bank_online import OnlineBankData
from .bank_addition import ContinualBankAddData, BankAddDataset
from .mushroom import ContinualMushroomData, MushroomDataset, OneHotMushroomDataset
from .mushroom_online import OnlineMushroomData
from .covertype import ContinualCoverData, CoverDataset, OneHotCoverDataset
from .covertype_online import OnlineCoverData
from .retail import ContinualRetailData, RetailDataset, OneHotRetailDataset
from .retail_online import OnlineRetailData, OnlineRetailDataset, OnlineOneHotRetailDataset
from .movielens import ContinualMovieLens, MovieLensDataset, OneHotMovieLensDataset


def dataloader(dataset_name, model_config, env_config):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    
    if dataset_name == 'adult':
        db = ContinualAdultData('./data/tabular_data/adult/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, AdultDataset
    elif dataset_name == 'adult_onehot':
        db = ContinualAdultData('./data/tabular_data/adult/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, OneHotAdultDataset
    if dataset_name == 'adult_online':
        db = OnlineAdultData('./data/tabular_data/adult/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, AdultDataset
    elif dataset_name == 'adult_onehot_online':
        db = OnlineAdultData('./data/tabular_data/adult/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, OneHotAdultDataset
    elif dataset_name == 'bank':
        db = ContinualBankData('./data/tabular_data/bank/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, BankDataset
    elif dataset_name == 'bank_onehot':
        db = ContinualBankData('./data/tabular_data/bank/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, OneHotBankDataset
    elif dataset_name == 'bank_online':
        db = OnlineBankData('./data/tabular_data/bank/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, BankDataset
    elif dataset_name == 'bank_onehot_online':
        db = OnlineBankData('./data/tabular_data/bank/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, OneHotBankDataset
    elif dataset_name == 'bank_add':
        db = ContinualBankAddData('./data/tabular_data/bank/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, BankAddDataset
    elif dataset_name == 'mushroom':
        db = ContinualMushroomData('./data/tabular_data/mushroom/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, MushroomDataset
    elif dataset_name == 'mushroom_onehot':
        db = ContinualMushroomData('./data/tabular_data/mushroom/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, OneHotMushroomDataset
    elif dataset_name == 'mushroom_online':
        db = OnlineMushroomData('./data/tabular_data/mushroom/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, MushroomDataset
    elif dataset_name == 'mushroom_onehot_online':
        db = OnlineMushroomData('./data/tabular_data/mushroom/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, OneHotMushroomDataset
    elif dataset_name == 'covertype':
        db = ContinualCoverData('./data/tabular_data/covertype/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, CoverDataset
    elif dataset_name == 'covertype_onehot':
        db = ContinualCoverData('./data/tabular_data/covertype/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, OneHotCoverDataset
    elif dataset_name == 'covertype_online':
        db = OnlineCoverData('./data/tabular_data/covertype/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, CoverDataset
    elif dataset_name == 'covertype_onehot_online':
        db = OnlineCoverData('./data/tabular_data/covertype/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, OneHotCoverDataset
    elif dataset_name == 'retail':
        db = ContinualRetailData('./data/time_tabular_data/retail/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, RetailDataset
    elif dataset_name == 'retail_onehot':
        db = ContinualRetailData('./data/time_tabular_data/retail/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, OneHotRetailDataset
    elif dataset_name == 'retail_online':
        db = OnlineRetailData('./data/time_tabular_data/retail/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, OnlineRetailDataset
    elif dataset_name == 'retail_onehot_online':
        db = OnlineRetailData('./data/time_tabular_data/retail/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, OnlineOneHotRetailDataset
    elif dataset_name == 'movielens':
        db = ContinualMovieLens('./data/movielens/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, MovieLensDataset
    elif dataset_name == 'movielens_onehot':
        db = ContinualMovieLens('./data/movielens/',
                             model_config=model_config,
                             env_config=env_config)
        
        return db, OneHotMovieLensDataset
    else:
        raise NotImplementedError()

    return train_dataset, val_dataset, test_dataset
