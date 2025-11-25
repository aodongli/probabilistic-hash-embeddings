import os
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


class MovieLensRecommendTrainer:
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

        self.exp_path = os.path.join(model_config['exp_path'], '%d' % int(time.time()))
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
        #     print(
        #         f"File {json_results} already present! Shutting down to prevent loss of previous experiments")


    def train_one_epoch(self, train_dataloader):
        self.model.train()
        loss_list, nll_list, kl_list, mae_list = [], [], [], []
        for minibatch in train_dataloader:
            cat = minibatch['cat'].to(self.device)
            movie_genre = minibatch['movie_genre'].to(self.device)
            target = minibatch['target'].float().to(self.device)
            dat_size = minibatch['dat_size'].float().to(self.device)
            pred, q_item, q_itemw, p_item, p_itemw = self.model(cat, movie_genre)
            nelbo, nll, kl, mae = self.loss(dat_size, pred, target,
                                            q_item, q_itemw, p_item, p_itemw)

            loss = nelbo.mean()
            loss_list.append(loss.detach().cpu().item())
            
            nll_list.append(nll.mean().detach().cpu().item())
            kl_list.append(kl.mean().detach().cpu().item())
            mae_list.append(mae.mean().detach().cpu().item())

            self.optimizer.zero_grad()
            loss.backward()
            # # debug info

            # # end
            self.optimizer.step()
            

        return [np.mean(loss_list), np.mean(nll_list), np.mean(kl_list), np.mean(mae_list)]


    def train(self, t, train_dataset, val_dataset=None, test_dataset=None):

        self.model.train()

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.model_config['batch_size'], 
                                      shuffle=True,
                                      drop_last=False)

        for epoch in range(self.model_config['epoch']):
            print(f'Epoch {epoch}:', end=' | ', flush=True)

            loss_epoch = self.train_one_epoch(train_dataloader)
            print(loss_epoch, flush=True)



    def train_init(self, train_dataset, val_dataset=None, test_dataset=None):

        self.model.train()

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.model_config['batch_size'], 
                                      shuffle=True,
                                      drop_last=False)
        
        for epoch in range(self.model_config['ini_task_epoch']):
            print(f'Epoch {epoch}:', end=' | '); sys.stdout.flush()

            loss_epoch = self.train_one_epoch(train_dataloader)
            print(loss_epoch, end=' | ', flush=True)

            if val_dataset is not None:
                res = self.test(val_dataset)
                print('val acc/nelbo:', res, end=' | ', flush=True)

            if test_dataset is not None:
                res = self.test(test_dataset)
                print('test acc/nelbo:', res, end='', flush=True)
            
            print('')

        torch.save(self.model.state_dict(), self.exp_path + '/init_model.pt')


    def set_prior(self):
        # for p in self.model.latent_hashemb.parameters():
        # for p in self.model.parameters():
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
        
        try:
            self.model.latent_hashemb.fill_prior(*list(self.model.latent_hashemb.parameters())[:4])
        except:
            self.model.latent_onehotemb.fill_prior(*list(self.model.latent_onehotemb.parameters())[:2])
        print('Load pre-trained prior')
        
    
    def train_scheduler(self, db, dataset_class, tot_task=286):
        print('Start training')
        # initialization
        init_dat, init_val, init_test = db.get_init_data()
        init_val_dataset = dataset_class(init_val, init_val[:, 2], db.movie_cat, dat_size=len(init_test), vocab_map=db.vocab_map)  # VALIDATION WITH dat_size
        print('initialization data size %d' % len(init_val_dataset))
        self.train_init(init_val_dataset, val_dataset=None, test_dataset=None)  # VALIDATION

        maes = []
        # online learning
        for t in range(tot_task):
            dat_tr, dat_val, dat_te = db.get_dataset(task_id=t)
            tr = dataset_class(dat_tr, dat_tr[:, 2], db.movie_cat, vocab_map=db.vocab_map)
            val = dataset_class(dat_val, dat_val[:, 2], db.movie_cat, dat_size=len(dat_te), vocab_map=db.vocab_map)  # VALIDATION WITH dat_size
            te = dataset_class(dat_te, dat_te[:, 2], db.movie_cat, vocab_map=db.vocab_map)
            
            dataset = val  # validation

            if len(dataset) == 0: continue

            # prediction
            res = self.test(dataset)
            print('before update:')

            print('\tmae', float(res[0]))
            print('\tnelbo', float(res[1]))
            print('\tnll', float(res[2]))

            maes.append(float(res[0]))

            self.set_prior()
            
            self.train(t, dataset)

        print('Result: %f' % (np.mean(maes)))
        print(maes)
        np.save(os.path.join(self.exp_path, 'mae.npy'), maes)
        return np.mean(maes)
            
    
    def test_scheduler(self, db, dataset_class, tot_task=286):
        res = None
        print('Start testing')
        # initialization
        if self.env_config.init_ckpt_path != '':
            self.model.load_state_dict(torch.load(self.env_config.init_ckpt_path))
            print('Load from pretrained models')
        else:
            init_dat, init_val, init_test = db.get_init_data()
            init_test_dataset = dataset_class(init_test, init_test[:, 2], db.movie_cat, vocab_map=db.vocab_map)
            init_val_dataset = dataset_class(init_val, init_val[:, 2], db.movie_cat, vocab_map=db.vocab_map)
            print('initialization data size (train=%d val=%d)' % (len(init_test_dataset), len(init_val_dataset)))
            self.train_init(init_test_dataset, val_dataset=init_val_dataset, test_dataset=None)

        maes = []
        # online learning
        for t in range(tot_task):
            dat_tr, dat_val, dat_te = db.get_dataset(task_id=t)
            te = dataset_class(dat_te, dat_te[:, 2], db.movie_cat, vocab_map=db.vocab_map)
            
            dataset = te  # test

            if len(dataset) == 0: continue

            # prediction
            res = self.test(dataset)
            print('before update:')

            print('\tmae', float(res[0]))
            print('\tnelbo', float(res[1]))
            print('\tnll', float(res[2]))

            maes.append(float(res[0]))

            self.set_prior()
            
            self.train(t, dataset)

        print('Result: %f' % (np.mean(maes)))
        print(maes)
        np.save(os.path.join(self.exp_path, 'mae.npy'), maes)
        return np.mean(maes)


    def test(self, test_dataset, ckpt_path=''):
        if ckpt_path != '':
            self.model.load_state_dict(torch.load(ckpt_path))

        self.model.eval()

        test_dataloader = DataLoader(test_dataset,
                                  batch_size=self.model_config['batch_size'], 
                                  shuffle=False,
                                  drop_last=False)

        preds, targets, nelbos, nlls, maes = [], [], [], [], []
        with torch.no_grad():
            for minibatch in test_dataloader:
                cat = minibatch['cat'].to(self.device)
                movie_genre = minibatch['movie_genre'].to(self.device)
                target = minibatch['target'].float().to(self.device)
                dat_size = minibatch['dat_size'].float().to(self.device)
                pred, q_item, q_itemw, p_item, p_itemw = self.model(cat, movie_genre)
                nelbo, nll, kl, mae = self.loss(dat_size, pred, target,
                                                q_item, q_itemw, p_item, p_itemw)

                # # debug info
                # # end

                preds.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())
                nelbos.append(nelbo.cpu().numpy())
                nlls.append(nll.cpu().numpy())
                maes.append(mae.cpu().numpy())

        mae = np.mean(maes)  # absolute error

        # if len(probs.shape) == 2:
        #     raise NotImplementedError
        return [str(mae), str(np.mean(nelbos)), str(np.mean(nlls))]