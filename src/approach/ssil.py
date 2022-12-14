import pdb

import torch
import warnings
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data import DataLoader, RandomSampler,Sampler
import torch.nn.functional as F
from collections import defaultdict

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from datasets.exemplars_selection import override_dataset_transform


class Appr(Inc_Learning_Appr):
    """
    """

    def __init__(self, model, device, nepochs=60, lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-5, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, ratio=4, distill_temp=2.0, lamb=1.0):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.exemplars_loader = None
        self.exemplars_iter = None
        self.ratio = ratio
        self.distill_temp = distill_temp
        self.lamb = lamb

        self.additional_stats = {'train': defaultdict(list), 'eval': defaultdict(list)}

        # SS-IL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: iCaRL is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 4. " allowing iCaRL to balance between CE and distillation loss."
        parser.add_argument('--ratio', default=4, type=float, required=False,
                            help="write-me") # TODO: write me
        parser.add_argument('--distill-temp', default=2., type=float, required=False,
                            help='Temperature to use in distillation softmax')
        parser.add_argument('--lamb', type=float, default=1., required=False,
                            help="")
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        self.current_epoch = 1

        # add exemplars to train_loader
        if t > 0:
            joint_loader = self.get_joint_loader(trn_loader)
            # TEST - dont use RP in exemplar selection
            sel_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        else:
            joint_loader = trn_loader
            sel_loader = trn_loader

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, joint_loader, val_loader)



        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, sel_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Save old model to extract features later. This is different from the original approach, since they propose to
        #  extract the features and store them for future usage. However, when using data augmentation, it is easier to
        #  keep the model frozen and extract the features when needed.
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def get_joint_loader(self, trn_loader):
        # balanced loader
        if self.ratio > 0:
            ex_batch_size = int(trn_loader.batch_size // self.ratio)

            self.batch_size = trn_loader.batch_size

            batch_sampler = BalancedSampler(sampler=RandomSampler(trn_loader.dataset),
                                            aux_n=len(self.exemplars_dataset),
                                            batch_size=trn_loader.batch_size,
                                            aux_batch_size=ex_batch_size)
            joint_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                       batch_sampler=batch_sampler,
                                                       num_workers=trn_loader.num_workers,
                                                       pin_memory=trn_loader.pin_memory)
        else:
            joint_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                       batch_size=trn_loader.batch_size,
                                                       shuffle=True,
                                                       num_workers=trn_loader.num_workers,
                                                       pin_memory=trn_loader.pin_memory)
        return joint_loader

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        outputs_old = None

        for images, targets in trn_loader:

            images = images.to(self.device)
            if t > 0:
                outputs_old = self.model_old(images)

            # Forward task data on current model
            outputs = self.model(images)
            # Calculate loss
            loss = self.criterion(t,
                                  outputs=outputs,
                                  targets=targets.to(self.device),
                                  outputs_old=outputs_old)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()


    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            outputs_old = None

            for images, targets in val_loader:
                # Forward old model
                images = images.to(self.device)
                if t > 0:
                    outputs_old = self.model_old(images)

                # Forward current model
                outputs = self.model(images)

                # during training, the usual accuracy is computed on the outputs
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log

                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)

                loss = self.criterion(t, outputs=outputs, targets=targets.to(self.device), outputs_old=outputs_old,
                                      training=False)

                total_loss += loss.item() * len(targets)


        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, outputs, targets, outputs_old=None, training=True):
        """Returns the loss value"""

        if training and t > 0:
            # current task loss
            loss = F.cross_entropy(outputs[t][:self.batch_size], targets[:self.batch_size] - self.model.task_offset[t])
            # exemplars loss
            loss += F.cross_entropy(torch.cat(outputs[:t], dim=1)[self.batch_size:], targets[self.batch_size:])
        else:
            loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)

        if t > 0 and outputs_old is not None:
            # Task-wise Knowledge Distillation loss on outputs for all previous tasks
            if self.lamb > 0:
                loss += self.lamb * sum(F.kl_div(F.log_softmax(p_t / self.distill_temp, dim=1),
                                                 F.log_softmax(q_t / self.distill_temp, dim=1),
                                                 reduction='batchmean',
                                                 log_target=True)
                                        for p_t, q_t in zip(outputs[:t], outputs_old[:t]))
        return loss


class BalancedSampler(Sampler):
    """A batch sampler that build mini batches from two datasets, first uses a sampler, second `aux` is suppose
    to be smaller and is reset then it ends. Used to implement Ratio Preserving loader for exemplars.
    """
    def __init__(self,
                 sampler: Sampler[int],
                 aux_n: int,
                 batch_size: int,
                 aux_batch_size: int) -> None:

        self.sampler = sampler
        self.main_size = len(sampler.data_source)
        self.batch_size = batch_size

        self.aux_n = aux_n
        self.aux_batch_size = aux_batch_size
        self.aux_ind = None

        self._reset_ex()

    def _reset_ex(self) -> None:
        self.ex_perm = torch.randperm(self.aux_n)
        self.aux_ind = 0

    def _get_ex_batch(self):
        # always drop last batch
        if self.aux_ind + self.aux_batch_size >= self.aux_n:
            self._reset_ex()

        ex_batch = self.ex_perm[self.aux_ind:(self.aux_ind + self.aux_batch_size)] + self.main_size
        self.aux_ind += self.aux_batch_size

        return ex_batch.tolist()

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch + self._get_ex_batch()
                batch = []

    def __len__(self) -> int:
        return self.main_size // self.batch_size

    @property
    def true_batch_size(self) -> int:
        return self.batch_size + self.aux_batch_size
