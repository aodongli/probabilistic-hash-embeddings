import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


class TimeDyTabPred_Trainer:
    def __init__(self, model_config, env_config):
        self.device = torch.device(model_config['device'])
        model_class = model_config['model_class']
        self.model = model_class(model_config, env_config).to(self.device)
        loss_class = model_config['loss_class']
        self.loss = loss_class(model_config, env_config).to(self.device)
        optim_class = model_config['optim_class']
        self.optimizer = optim_class(self.model.parameters(),
                                     lr=model_config['learning_rate'], 
                                     weight_decay=model_config['l2'])

        self.model_config = model_config
        self.env_config = env_config
        
        self.optim_class = optim_class
        self.loss_class = loss_class

        self.cat_nograd = False

        self.seq_len = self.model_config['sample_seq_len']

        self.exp_path = model_config['exp_path']
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
        #     print(
        #         f"File {json_results} already present! Shutting down to prevent loss of previous experiments")


    def train_one_epoch(self, train_dataloader):
        self.model.train()
        loss_list, nll_list, kl_list = [], [], []
        for minibatch in train_dataloader:
            obs = minibatch['obs'].to(self.device)
            cat = minibatch['cat'].to(self.device)
            cat_incre = minibatch['cat_incre'].to(self.device)
            target = minibatch['target'].float().to(self.device)
            time = minibatch['time'].float().to(self.device)
            dat_size = minibatch['dat_size'].float().to(self.device)
            rate, q_item, q_itemw, q_time, p_item, p_itemw, p_time = self.model(cat, cat_incre, obs, target, time, cat_nograd=self.cat_nograd)
            nelbo, nll, kl = self.loss(dat_size, rate, target,
                              q_item, q_itemw, q_time, p_item, p_itemw, p_time)

            loss = nelbo.mean()
            loss_list.append(loss.detach().cpu().item())
            
            nll_list.append(nll.mean().detach().cpu().item())
            kl_list.append(kl.mean().detach().cpu().item())

            self.optimizer.zero_grad()
            loss.backward()
            # # debug info

            # # end
            self.optimizer.step()

        return [np.mean(loss_list), np.mean(nll_list), np.mean(kl_list)]


    def train(self, t, train_dataset, val_dataset=None, test_dataset=None):

        self.model.train()

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.model_config['batch_size'], 
                                      shuffle=True,
                                      drop_last=False)
        
        num_epoch = self.model_config['ini_task_epoch'] if t == 0 else self.model_config['epoch']

        for epoch in range(num_epoch):
            print(f'Epoch {epoch}:', end=' | ')

            loss_epoch = self.train_one_epoch(train_dataloader)
            print(loss_epoch, end=' | ')

            if val_dataset is not None:
                res = self.test(val_dataset)
                print('val acc/nelbo:', res, end=' | ')

            if test_dataset is not None:
                res = self.test(test_dataset)
                print('test acc/nelbo:', res)

        torch.save(self.model.state_dict(), self.exp_path + '/model.pt')
        
    
    def train_scheduler(self, db, dataset_class, tot_task=3):
        continual_res = []
        continual_res_2 = []
        continual_res_3 = []
        for t in range(tot_task):
            dat_tr, dat_val, dat_te = db.get_dataset(task_id=t, task_num=tot_task)
            tr = dataset_class(dat_tr[:, :-1], dat_tr[:, -1], db.incre_col_idx, db.cat_id, seq_len=self.seq_len)
            val = dataset_class(dat_val[:, :-1], dat_val[:, -1], db.incre_col_idx, db.cat_id, seq_len=self.seq_len)
            te = dataset_class(dat_te[:, :-1], dat_te[:, -1], db.incre_col_idx, db.cat_id, seq_len=self.seq_len)
            print('before update:', self.test(te))
            if t > 0:
                try:
                    self.optimizer = self.optim_class(
                        self.model.latent_hashemb.parameters(),
                        lr=self.model_config['learning_rate'], 
                        weight_decay=self.model_config['l2'])
                except:
                    self.optimizer = self.optim_class(
                        self.model.latent_onehotemb.parameters(),
                        lr=self.model_config['learning_rate'], 
                        weight_decay=self.model_config['l2'])
                print('Update optimizer')

                self.cat_nograd = True
                if self.cat_nograd:
                    print('Do not update embeddings other than incremental column')
                
                try:
                    self.model.latent_hashemb.fill_prior(*list(self.model.latent_hashemb.parameters())[:4])
                except:
                    self.model.latent_onehotemb.fill_prior(*list(self.model.latent_onehotemb.parameters())[:2])
                print('Load pre-trained prior')
            self.train(t, tr, val_dataset=val, test_dataset=te)
            
            print('Test:')
            task_res = [0.] * tot_task
            task_res_2 = [0.] * tot_task
            task_res_3 = [0.] * tot_task
            for _t in range(t+1):
                _, _, dat_te = db.get_dataset(task_id=_t, task_num=tot_task)
                te = dataset_class(dat_te[:, :-1], dat_te[:, -1], db.incre_col_idx, db.cat_id, seq_len=self.seq_len)
                res = self.test(te)
                print(res)
                task_res[_t] = float(res[0])
                task_res_2[_t] = float(res[1])
                task_res_3[_t] = float(res[2])
            continual_res.append(task_res)
            continual_res_2.append(task_res_2)
            continual_res_3.append(task_res_3)
        print('mae', continual_res)
        print('mce', continual_res_2)
        print('nelbo', continual_res_3)
        
    
    def global_train_scheduler(self, db, dataset_class, tot_task=3):
        dat_tr_list, dat_val_list, dat_te_list = [], [], []
        for t in range(tot_task):
            dat_tr, dat_val, dat_te = db.get_dataset(task_id=t, task_num=tot_task)
            dat_tr_list.append(dat_tr)
            dat_val_list.append(dat_val)
            dat_te_list.append(dat_te)
        tot_dat_tr = np.concatenate(dat_tr_list)
        tot_dat_val = np.concatenate(dat_val_list)
        tot_dat_te = np.concatenate(dat_te_list)
        tr = dataset_class(tot_dat_tr[:, :-1], tot_dat_tr[:, -1], db.incre_col_idx, db.cat_id, seq_len=self.seq_len)
        val = dataset_class(tot_dat_val[:, :-1], tot_dat_val[:, -1], db.incre_col_idx, db.cat_id, seq_len=self.seq_len)
        te = dataset_class(tot_dat_te[:, :-1], tot_dat_te[:, -1], db.incre_col_idx, db.cat_id, seq_len=self.seq_len)
        self.train(0, tr, val_dataset=val, test_dataset=te)
    
    
    def test_scheduler(self, db, dataset_class, tot_task=3):
        print('Final test:')
        for t in range(tot_task):
            dat_tr, dat_val, dat_te = db.get_dataset(task_id=t, task_num=tot_task)
            te = dataset_class(dat_te[:, :-1], dat_te[:, -1], db.incre_col_idx, db.cat_id, seq_len=self.seq_len)
            res = self.test(te)
            print(res)
        return res


    def test(self, test_dataset, ckpt_path=''):
        if ckpt_path != '':
            self.model.load_state_dict(torch.load(ckpt_path))

        self.model.eval()

        test_dataloader = DataLoader(test_dataset,
                                  batch_size=self.model_config['batch_size'], 
                                  shuffle=False,
                                  drop_last=False)

        rates, targets, nelbos, nlls = [], [], [], []
        with torch.no_grad():
            for minibatch in test_dataloader:
                obs = minibatch['obs'].to(self.device)
                cat = minibatch['cat'].to(self.device)
                cat_incre = minibatch['cat_incre'].to(self.device)
                target = minibatch['target'].float().to(self.device)
                time = minibatch['time'].float().to(self.device)
                dat_size = minibatch['dat_size'].float().to(self.device)
                rate, q_item, q_itemw, q_time, p_item, p_itemw, p_time = self.model(cat, cat_incre, obs, target, time, cat_nograd=self.cat_nograd)
                nelbo, nll, kl = self.loss(dat_size, rate, target,
                                  q_item, q_itemw, q_time, p_item, p_itemw, p_time)

                # # debug info
                # # end

                rates.append(rate.cpu().numpy())
                targets.append(target.cpu().numpy())
                nelbos.append(nelbo.cpu().numpy())
                nlls.append(nll.cpu().numpy())

        rates = np.concatenate(rates)
        targets = np.concatenate(targets)
        mae = np.mean(np.abs(rates[:,-1] - targets[:,-1]))  # absolute error
        mce = np.mean(np.abs(rates - targets))  # cumulative error

        # if len(probs.shape) == 2:
        #     raise NotImplementedError
        return [str(mae), str(mce), str(np.mean(nelbos)), str(np.mean(nlls))]