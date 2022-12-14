import torch
import math
import time
import numpy as np
import warnings
from copy import deepcopy
from argparse import ArgumentParser
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .LAS_utils import mixup_data, mixup_criterion,LabelAwareSmoothing, LearnableWeightScaling
import  datasets.data_loader as stage2_utils

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the End-to-end Incremental Learning (EEIL) approach described in
    http://openaccess.thecvf.com/content_ECCV_2018/papers/Francisco_M._Castro_End-to-End_Incremental_Learning_ECCV_2018_paper.pdf
    Original code available at https://github.com/fmcp/EndToEndIncrementalLearning
    Helpful code from https://github.com/arthurdouillard/incremental_learning.pytorch
    """

    def __init__(self, model, device, nepochs=90, lr=0.1, lr_min=1e-6, lr_factor=10, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.0001, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, lamb=1.0, T=2, lr_finetuning_factor=0.1,
                 nepochs_finetuning=40, noise_grad=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.lr_finetuning_factor = lr_finetuning_factor
        self.nepochs_finetuning = nepochs_finetuning
        self.noise_grad = noise_grad
        self.lws_models = torch.nn.ModuleList()
        self.stage2epoch = 30
        self._train_epoch = 0
        self._finetuning_balanced = None

        # EEIL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: EEIL is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    # @staticmethod
    def _get_optimizer(self,stage2=False):
        """Returns the optimizer"""
        if stage2:
            params = list(self.model.heads[-1].parameters())
            return torch.optim.SGD([{"params":params},{"params":self.lws_models.parameters()}], 
            self.lr, weight_decay=self.wd, momentum=self.momentum)           
        else:
            return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Added trade-off between the terms of Eq. 1 -- L = L_C + lamb * L_D
        parser.add_argument('--lamb', default=1.0, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Page 6: "Based on our empirical results, we set T to 2 for all our experiments"
        parser.add_argument('--T', default=2.0, type=float, required=False,
                            help='Temperature scaling (default=%(default)s)')
        # "The same reduction is used in the case of fine-tuning, except that the starting rate is 0.01."
        parser.add_argument('--lr-finetuning-factor', default=0.01, type=float, required=False,
                            help='Finetuning learning rate factor (default=%(default)s)')
        # Number of epochs for balanced training
        parser.add_argument('--nepochs-finetuning', default=40, type=int, required=False,
                            help='Number of epochs for balanced training (default=%(default)s)')
        # the addition of noise to the gradients
        parser.add_argument('--noise-grad', action='store_true',
                            help='Add noise to gradients (default=%(default)s)')
        return parser.parse_known_args(args)

    def _train_unbalanced(self, t, trn_loader, val_loader, train= True):
        """Unbalanced training"""
        self._finetuning_balanced = False
        self._train_epoch = 0
        loader = self._get_train_loader(trn_loader, False)
        if train:
            super().train_loop(t, loader, val_loader)
        return loader

    def _train_balanced(self, t, trn_loader, val_loader):
        """Balanced finetuning"""
        self._finetuning_balanced = True
        self._train_epoch = 0
        orig_lr = self.lr
        self.lr *= self.lr_finetuning_factor
        orig_nepochs = self.nepochs
        self.nepochs = self.nepochs_finetuning
        loader = self._get_train_loader(trn_loader, True)
        super().train_loop(t, loader, val_loader)
        self.lr = orig_lr
        self.nepochs = orig_nepochs

    def _get_train_loader(self, trn_loader, balanced=False):
        """Modify loader to be balanced or unbalanced"""
        exemplars_ds = self.exemplars_dataset
        trn_dataset = trn_loader.dataset
        if balanced:
            indices = torch.randperm(len(trn_dataset))
            trn_dataset = torch.utils.data.Subset(trn_dataset, indices[:len(exemplars_ds)])
        ds = exemplars_ds + trn_dataset
        return DataLoader(ds, batch_size=trn_loader.batch_size,
                              shuffle=True,
                              num_workers=trn_loader.num_workers,
                              pin_memory=trn_loader.pin_memory)

    def pre_train_process(self, t, trn_loader):
        lws_model = LearnableWeightScaling(num_classes=self.model.task_cls[t]).to(self.device)
        self.lws_models.append(lws_model)
        super().pre_train_process(t, trn_loader)

    def _noise_grad(self, parameters, iteration, eta=0.3, gamma=0.55):
        """Add noise to the gradients"""
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        variance = eta / ((1 + iteration) ** gamma)
        for p in parameters:
            p.grad.add_(torch.randn(p.grad.shape, device=p.grad.device) * variance)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader2 = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                batch_size=trn_loader.batch_size,
                                                shuffle=True,
                                                num_workers=trn_loader.num_workers,
                                                pin_memory=trn_loader.pin_memory)
        if t == 0:  # First task is simple training
            super().train_loop(t, trn_loader, val_loader)
            # modelstate = torch.load('modeltask0eeil.pt')
            # self.model.load_state_dict(modelstate)
            loader = trn_loader
        else:
            # Page 4: "4. Incremental Learning" -- Only modification is that instead of preparing examplars before
            # training, we do it online using the stored old model.

            # Training process (new + old) - unbalanced training
            loader = self._train_unbalanced(t, trn_loader, val_loader)
            # Balanced fine-tunning (new + old)
            self._train_balanced(t, trn_loader, val_loader)

        # After task training： update exemplars
        self.exemplars_dataset.collect_exemplars(self.model, loader, val_loader.dataset.transform)
        if t != 0:
            balance_sampler = stage2_utils.ClassAwareSampler(trn_loader2.dataset)
            balanced_trn_loader = DataLoader(trn_loader2.dataset,
                                    batch_size=trn_loader.batch_size,
                                    shuffle=False,
                                    num_workers=trn_loader.num_workers,
                                    pin_memory=trn_loader.pin_memory,
                                    sampler=balance_sampler)
        else:
            balance_sampler = stage2_utils.ClassAwareSampler(trn_loader.dataset)
            balanced_trn_loader = DataLoader(trn_loader.dataset,
                                    batch_size=trn_loader.batch_size,
                                    shuffle=False,
                                    num_workers=trn_loader.num_workers,
                                    pin_memory=trn_loader.pin_memory,
                                    sampler=balance_sampler)
        # balance_sampler = stage2_utils.ClassAwareSampler(self.exemplars_dataset)
        # balanced_trn_loader = DataLoader(self.exemplars_dataset,
        #                         batch_size=trn_loader.batch_size,
        #                         shuffle=False,
        #                         num_workers=trn_loader.num_workers,
        #                         pin_memory=trn_loader.pin_memory,
        #                         sampler=balance_sampler)
            # num_samples = self.get_data_distribution(trn_loader.dataset)
        """stage2"""
        lr=self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()
        self.optimizer = self._get_optimizer(stage2=True)
        # for e in range(self.stage2epoch):
        for e in range(1):
        # Train
            clock0 = time.time()
            self.train_epoch_stage2(t, balanced_trn_loader)
            # self.train_epoch_stage2(t, trn_loader)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |stage2'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |stage2'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = 10
                print(' *', end='')
            self.adjust_learning_rate_stage_2(e+1)
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()

    def train_epoch_stage2(self , t,train_loader):
        training_data_num = len(train_loader.dataset)
        end_steps = int(np.ceil(float(training_data_num) / float(train_loader.batch_size)))

        # switch to train mode
        # print(self.model.model)
        # print(self.model.heads)
        self.model.model.eval()
        self.model.heads.train()
        for i, (images, target) in enumerate(train_loader):
            if i > end_steps:
                break
            with torch.no_grad():
                feat = self.model.model(images.to(self.device))
            # feat = self.model.model(images.to(self.device))
            output=[]
            for idx in range(len(self.model.heads)):
                output.append(self.model.heads[idx](feat.detach()))
                # output.append(self.model.heads[idx](feat))
                # print(output[idx])
                output[idx] = self.lws_models[idx](output[idx])
            # output = lws_model(output)
            # ref_outputs = None
            # ref_features = None
            # if t > 0:
            #     ref_outputs, ref_features = self.ref_model(images, return_features=True)
            loss = self.criterion(t,output, target.to(self.device),stage2 = True)
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def adjust_learning_rate_stage_2(self,epoch):
        """Sets the learning rate"""
        lr_min = 0
        lr_max = self.lr
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / self.stage2epoch * 3.1415926535))
        print(' lr={:.1e}'.format(lr), end='')
        for idx, param_group in enumerate(self.optimizer.param_groups):
            if idx == 1:
                param_group['lr'] = (1/self.lr_factor) * lr
            else:
                param_group['lr'] = 1.00 * lr

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Save old model to extract features later
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images = images.to(self.device)
            # Forward old model
            outputs_old = None
            if t > 0:
                outputs_old = self.model_old(images)
            # Forward current model
            outputs = self.model(images)
            loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # Page 8: "We apply L2-regularization and random noise [21] (with parameters eta = 0.3, gamma = 0.55)
            # on the gradients to minimize overfitting"
            # https://github.com/fmcp/EndToEndIncrementalLearning/blob/master/cnn_train_dag_exemplars.m#L367
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            if self.noise_grad:
                self._noise_grad(self.model.parameters(), self._train_epoch)
            self.optimizer.step()
        self._train_epoch += 1

    def criterion(self, t, outputs, targets, outputs_old=None,stage2=False):
        """Returns the loss value"""
        if stage2:
            outputs = torch.cat(outputs, dim=1)
            loss = torch.nn.CrossEntropyLoss(None)(outputs, targets)
        else:
            # Classification loss for new classes
            loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
            # Distilation loss
            if t > 0 and outputs_old:
                # take into account current head when doing balanced finetuning
                last_head_idx = t if self._finetuning_balanced else (t - 1)
                for i in range(last_head_idx):
                    loss += self.lamb * F.binary_cross_entropy(F.softmax(outputs[i] / self.T, dim=1),
                                                            F.softmax(outputs_old[i] / self.T, dim=1))
        return loss

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                # if t!=0:
                for idx in range(len(outputs)):
                    # output.append(self.model.heads[idx](feat.detach()))
                    outputs[idx] = self.lws_models[idx](outputs[idx])
                loss = self.criterion(t, outputs, targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num
