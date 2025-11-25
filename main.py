import argparse
import os
import json
import numpy as np
from data_loader.data_loader import dataloader
from config.parser import model_config_reader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file', default='config_adult.yml')
    parser.add_argument('--dataset-name', dest='dataset_name', default='adult')
    parser.add_argument('--reg-weight', dest='reg_weight', type=float, default=1.0)
    parser.add_argument('--incre-col', dest='incre_col', default='education', help='Not used in the recommendation task')
    parser.add_argument('--init-ckpt-path', dest='init_ckpt_path', default='')
    parser.add_argument('--cat-grad', dest='cat_grad', default=False, action='store_true', help='If False, do not update embeddings other than incremental columns.')
    parser.add_argument('--random-task-size', dest='random_task_size', default=False, action='store_true')
    parser.add_argument('--noise-obs', dest='noise_obs', default=False, action='store_true')
    return parser.parse_args()


def run_dataset(env_config, model_config):
    res = None
    trainer_class = model_config['trainer_class']
    trainer = trainer_class(model_config, env_config)
    db, CustomDataset = dataloader(env_config.dataset_name, model_config, env_config)
    res = trainer.test_scheduler(db, CustomDataset, tot_task=model_config['num_task'])
    model_config['result'] = res
    # save configs
    del model_config['model_class']
    del model_config['loss_class']
    del model_config['optim_class']
    del model_config['trainer_class']
    with open(model_config['exp_path']+'/model_config.json', 'w') as f:
        json.dump(model_config, f)
    with open(model_config['exp_path']+'/env_config.json', 'w') as f:
        json.dump(env_config.__dict__, f)
    print('result:', res)
    return res


if __name__ == "__main__":
    env_config = get_args()
    model_config = model_config_reader(env_config.config_file)
    
    res = run_dataset(env_config, model_config)