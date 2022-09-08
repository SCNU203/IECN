from __future__ import absolute_import

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from reid.lib.normalize import Normalize


__all__ = ['FtResNet', 'ftresnet18', 'ftresnet34', 'ftresnet50', 'ftresnet101',
           'ftresnet152']


class FtResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, num_triplet_features=0, n_splits=10, batch_size=128, adjustment=None):
        super(FtResNet, self).__init__()
        # split_conv_out_channels = num_features // n_splits
        split_conv_out_channels = num_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.adjustment = adjustment
        self.split_conv_out_channels = split_conv_out_channels

        # Construct base (pretrained) resnet
        if depth not in FtResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = FtResNet.__factory[depth](pretrained=pretrained)

        # Fix layers [conv1 ~ layer2]
        fixed_names = []
        for name, module in self.base._modules.items():
            if name == "layer3":
                # assert fixed_names == ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"]
                break
            fixed_names.append(name)
            for param in module.parameters():
                param.requires_grad = False

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            self.num_triplet_features = num_triplet_features

            self.l2norm = Normalize(2)

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
                init.constant_(self.feat_bn.weight, 1)
                init.constant_(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            self.split_conv_list = nn.ModuleList()
            self.split_bn_list = nn.ModuleList()
            for _ in range(self.n_splits):
                self.split_conv_list.append(nn.Sequential(
                    nn.Conv2d(2048, split_conv_out_channels, 1),
                    nn.BatchNorm2d(split_conv_out_channels),
                    nn.ReLU(inplace=True)
                ))
            if self.dropout >= 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)

            # feature-wise Adjustment
            # if self.adjustment == 'feature-wise':
            self.feat_wise_classifiers = nn.ModuleList()
            for _ in range(self.n_splits):
                fc = nn.Linear(self.split_conv_out_channels, self.num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.feat_wise_classifiers.append(fc)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, output_feature=None):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            else:
                x = module(x)

        if self.cut_at_pooling:
            return x

        if output_feature == 'feature-wise':
            split_h = x.size(2) / self.n_splits
            # splits_precs[n_splits, 128, 702]
            splits_precs = []
            for i in range(self.n_splits):
                split_feat = F.avg_pool2d(
                    x[:, :, int(i * split_h): int((i + 1) * split_h), :],
                    (int((i + 1) * split_h) - int(i * split_h), x.size(-1)))

                split_feat = self.split_conv_list[i](split_feat)
                split_feat = split_feat.view(split_feat.size(0), -1)

                # feature-wise Adjustment
                if output_feature == 'feature-wise':
                    # split_feat = F.relu(split_feat)
                    split_feat = self.drop(split_feat)
                    prec = self.feat_wise_classifiers[i](split_feat)
                    splits_precs.append(prec)
            return splits_precs

        if self.has_embedding and output_feature == 'tgt_feat':
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(x.size(0), -1)
            x = self.feat(x)
            x = self.feat_bn(x)
            tgt_feat = F.normalize(x)
            tgt_feat = self.drop(tgt_feat)
            return tgt_feat

        if output_feature == 'pool5':
            x = F.normalize(x)
            return x

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
            tgt_feat = F.normalize(x)
            tgt_feat = self.drop(tgt_feat)
            if output_feature == 'tgt_feat':
                return tgt_feat
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def ftresnet18(**kwargs):
    return FtResNet(18, **kwargs)


def ftresnet34(**kwargs):
    return FtResNet(34, **kwargs)


def ftresnet50(**kwargs):
    return FtResNet(50, **kwargs)


def ftresnet101(**kwargs):
    return FtResNet(101, **kwargs)


def ftresnet152(**kwargs):
    return FtResNet(152, **kwargs)