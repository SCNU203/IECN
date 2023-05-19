from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
from tqdm import  tqdm

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
import copy
import numpy as np
import visdom
import os
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, model, model_inv, lmd=0.5, n_splits=8, adjustment='feature-wise', num_classes=0, num_features=0):
        super(Trainer, self).__init__()
        self.n_splits = n_splits
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model_inv = model_inv
        self.pid_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.lmd = lmd
        self.adjustment = adjustment
        self.num_classes = num_classes
        self.num_features = num_features
        self.mean_feat = None

    def get_all_mean_feat(self, data_loader):
        with torch.no_grad():
            self.all_mean_feat = torch.zeros(self.n_splits, self.num_classes, self.num_features).cuda()
            count = torch.zeros(self.num_classes).cuda()
            pbar = tqdm(data_loader)
            for i, inputs in enumerate(pbar):
                inputs, pids = self._parse_data(inputs)
                # feats[n_splits, 128, 2048]
                feats = self.model(inputs, 'mean_feat')
                for j in range(feats.size(0)):
                    for k in range(feats.size(1)):
                        if j == 0:
                            count[pids[k]] = count[pids[k]] + 1
                        self.all_mean_feat[j, pids[k]] = self.all_mean_feat[j, pids[k]] + feats[j, k]

                pbar.set_description("Get All Mean Feat")
            self.all_mean_feat = self.all_mean_feat / count.view(1, -1, 1)
        return self.all_mean_feat

    def train(self, epoch, data_loader, target_train_loader, optimizer, print_freq=1):
        self.set_model_train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        # Target iter
        target_iter = iter(target_train_loader)
        mean_feat = None
        if self.adjustment == 'class-wise' or self.adjustment == 'Combined':
            mean_feat = self.get_all_mean_feat(data_loader)

        # Train
        pbar = tqdm(data_loader)
        for i, inputs in enumerate(pbar):
            data_time.update(time.time() - end)

            # Source inputs
            inputs, pids = self._parse_data(inputs)

            # Target inputs
            try:
                inputs_target = next(target_iter)
            except:
                target_iter = iter(target_train_loader)
                inputs_target = next(target_iter)
            inputs_target, index_target = self._parse_tgt_data(inputs_target)

            outputs = None
            if self.adjustment == 'feature-wise':
                outputs = self.model(inputs, self.adjustment)
            elif self.adjustment == 'class-wise' and epoch > 0:
                outputs = self.model(inputs, self.adjustment, mean_feat=mean_feat)
            elif self.adjustment == 'Combined' and epoch == 0:
                outputs = self.model(inputs, self.adjustment)
            elif self.adjustment == 'Combined' and epoch > 0:
                outputs = self.model(inputs, self.adjustment, mean_feats=mean_feat)
            else:
                outputs = self.model(inputs)
            source_pid_loss = 0
            prec1 = 0
            if self.adjustment == 'feature-wise' or self.adjustment == 'Combined':
                for j in range(self.n_splits):
                    source_pid_loss_item = self.pid_criterion(outputs[j], pids)
                    source_pid_loss = source_pid_loss + source_pid_loss_item
                source_pid_loss /= self.n_splits
                sum_prec = torch.sum(F.softmax(torch.stack(outputs), dim=2), dim=0) / self.n_splits
                prec, = accuracy(sum_prec.data, pids.data)
                prec1 = prec[0]
            else:
                source_pid_loss = self.pid_criterion(outputs, pids)
                prec, = accuracy(outputs.data, pids.data)
                prec1 = prec[0]

            # Target invariance loss
            outputs = self.model(inputs_target, 'tgt_feat')
            alpha_loss, beta_loss = self.model_inv(outputs, index_target, epoch=epoch)
            if epoch >= 5:
                loss_un = alpha_loss + beta_loss
            else:
                loss_un = beta_loss

            loss = (1 - self.lmd) * source_pid_loss + self.lmd * loss_un

            loss_print = {}
            loss_print['s_pid_loss'] = source_pid_loss.item()
            loss_print['t_alpha_loss'] = alpha_loss.item()
            loss_print['t_beta_loss'] = beta_loss.item()

            losses.update(loss.item(), outputs.size(0))
            precisions.update(prec1, outputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                log = "Epoch: [{}], Time {:.3f}, Loss {:.3f}, Prec {:.2%}" \
                    .format(epoch,
                            batch_time.sum,
                            losses.val,
                            precisions.val)

                for tag, value in loss_print.items():
                    log += ", {}: {:.4f}".format(tag, value)
                pbar.set_description("%s"%(log))

        log = "Epoch: [{}], Time {:.3f} ({:.3f}), Data {:.3f} ({:.3f}), Loss {:.3f} ({:.3f}), Prec {:.2%} ({:.2%})" \
            .format(epoch,
                    batch_time.sum, batch_time.avg,
                    data_time.sum, data_time.avg,
                    losses.val, losses.avg,
                    precisions.val, precisions.avg)
        for tag, value in loss_print.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        return inputs, pids

    def _parse_tgt_data(self, inputs_target):
        inputs, _, _, index = inputs_target
        inputs = inputs.to(self.device)
        index = index.to(self.device)
        return inputs, index

    def set_model_train(self):
        self.model.train()

        # Fix first BN
        fixed_bns = []
        for idx, (name, module) in enumerate(self.model.module.named_modules()):
            if name.find("layer3") != -1:
                # assert len(fixed_bns) == 22
                break
            if name.find("bn") != -1:
                fixed_bns.append(name)
                module.eval()
