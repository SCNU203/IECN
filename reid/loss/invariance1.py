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
    def __init__(self, num_features, num_classes, beta=0.05, knn=6, alpha=0.01, n_splits=10):
        super(InvNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = alpha  # Memory update rate
        self.beta = beta  # Temperature fact
        self.knn = knn  # Knn for neighborhood invariance
        self.n_splits = n_splits

        # Exemplar memory
        self.em = nn.Parameter(torch.zeros(num_classes, num_features), requires_grad=False)

    def forward(self, inputs, targets, epoch=None):

        alpha = self.alpha * epoch
        # sim[128,12936]
        sim = ExemplarMemory.apply(inputs, targets, self.em, alpha)
        em = self.em

        _, feats = em.size()
        step = feats // self.n_splits
        if step * self.n_splits < feats:
            step += 1
        sims = torch.Tensor(self.n_splits, inputs.size(0), sim.size(1)).to(self.device)
        for i in range(self.n_splits):
            l = int(step * i)
            r = int(step * (i + 1))
            sims[i] = torch.tensor(torch.tensor(inputs[:, l:r]).mm(torch.tensor(em[:, l:r]).t())).to(self.device)

        sim /= self.beta
        sims /= self.beta
        if self.knn > 0 and epoch > 4:
            # With neighborhood invariance
            loss = self.smooth_loss(sim, sims, targets)
        else:
            # Without neighborhood invariance
            loss = F.cross_entropy(sim, targets)
        return loss

    def smooth_loss(self, inputs, sims, targets):
        # target[n_splits, 128, 12936]
        targets = self.smooth_hot(sims.detach().clone(), targets.detach().clone(), self.knn)
        # outputs[128, 12936]
        outputs = F.log_softmax(inputs, dim=1)
        loss = 0
        for i in range(self.n_splits):
            splits_loss = - (targets[i] * outputs)
            splits_loss = splits_loss.sum(dim=1)
            splits_loss = splits_loss.mean(dim=0)
            loss = loss + splits_loss
        loss = loss / self.n_splits
        return loss

    # inputs[n_splits, 128, 12936] targets_onehots[n_splits, 128, 12936] targets[128]
    def smooth_hot(self, inputs, targets, k=6):
        # Sort
        _, index_sorted = torch.sort(inputs, dim=2, descending=True)
        targets_onehots = []
        targets = torch.unsqueeze(targets, 1)
        for i in range(self.n_splits):
            targets_onehot = torch.zeros(inputs[0].size()).to(self.device)

            ones_mat = torch.ones(targets.size(0), k).to(self.device)
            weights = F.softmax(ones_mat, dim=1)
            targets_onehot.scatter_(1, index_sorted[i, :, 0:k], ones_mat * weights)
            targets_onehot.scatter_(1, targets, float(1))
            targets_onehots.append(targets_onehot)

        return torch.stack(targets_onehots)


