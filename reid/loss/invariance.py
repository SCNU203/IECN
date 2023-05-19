import time

import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np
import math


class ExemplarMemory(Function):
    # def __init__(self, em, alpha=0.01):
    #     super(ExemplarMemory, self).__init__()
    #     self.em = em
    #     self.alpha = alpha

    @staticmethod
    def forward(ctx, inputs, targets, em, alpha=0.01):
        ctx.save_for_backward(inputs, targets)
        ctx.em = em
        ctx.alpha = alpha
        outputs = inputs.mm(em.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.em)
        for x, y in zip(inputs, targets):
            ctx.em[y] = ctx.alpha * ctx.em[y] + (1. - ctx.alpha) * x
            ctx.em[y] /= ctx.em[y].norm()
        return grad_inputs, None, None, None

# Invariance learning loss
class InvNet(nn.Module):
    def __init__(self, num_features, num_classes, beta=0.05, knn=6, alpha=0.01, n_splits=10, ul_alpha=0.05, ul_beta=1.0):
        super(InvNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = alpha  # Memory update rate
        self.beta = beta  # Temperature fact
        self.knn = knn  # Knn for neighborhood invariance
        self.n_splits = n_splits
        self.ul_alpha = ul_alpha
        self.ul_beta = ul_beta

        # Exemplar memory
        self.em = nn.Parameter(torch.zeros(num_classes, num_features))

    def forward(self, inputs, targets, epoch=None):

        alpha = self.alpha * epoch
        # memory = ExemplarMemory(self.em, alpha=alpha)
        inputs = ExemplarMemory.apply(inputs, targets, self.em, alpha)

        inputs /= self.beta
        if self.knn > 0 and epoch >= 5:
            targets = self.smooth_hot(inputs.detach().clone(), targets.detach().clone(), self.knn)

            # With neighborhood invariance
            # targets_re[targets_re == 0.0].data = -1e-4
            beta_loss = self.smooth_loss(inputs, targets)
            # With Symmetric Cross Entropy
            targets_re = targets.detach().clone()
            targets_re_mask = torch.zeros(targets_re.size()).cuda()
            targets_re_mask += 1e-4
            targets_re = torch.where(targets_re == 0.0, targets_re_mask, targets_re)
            alpha_loss = - (F.softmax(inputs,dim=1) * torch.log(targets_re))
            alpha_loss = alpha_loss.sum(dim=1)
            alpha_loss = alpha_loss.mean(dim=0)
            # loss = 0.05 * alpha_loss + beta_loss
            return self.ul_alpha * alpha_loss, self.ul_beta * beta_loss
        else:
            # Without neighborhood invariance
            loss = F.cross_entropy(inputs, targets)
            return torch.tensor([0]), loss
        return torch.tensor([0]), loss

    def smooth_loss(self, inputs, targets):

        outputs = F.log_softmax(inputs, dim=1)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def smooth_hot(self, inputs, targets, k=6):
        # Sort
        _, index_sorted = torch.sort(inputs, dim=1, descending=True)
        m, n = index_sorted.size()

        ones_mat = torch.ones(targets.size(0), k).to(self.device)
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.zeros(inputs.size()).to(self.device)
        for i in range(m):
            k_reciprocal_neigh = self.get_k_reciprocal_neigh(index_sorted, i, k)
            weights = 3.0/k
            targets_onehot[i, k_reciprocal_neigh] = weights

        targets_onehot.scatter_(1, targets, float(1))
        return targets_onehot

    def get_k_reciprocal_neigh(self, index_sorted, i, k):
        k_neigh_idx = index_sorted[i, :k]
        neigh_sims = self.em[k_neigh_idx].mm(self.em.t())

        _, neigh_idx_sorted = torch.sort(neigh_sims.detach().clone(), dim=1, descending=True)
        fi = np.where(neigh_idx_sorted[:, :k].cpu() == index_sorted.cpu()[i, 0])[0]
        return index_sorted[i, fi]