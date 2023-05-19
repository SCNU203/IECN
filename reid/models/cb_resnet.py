from __future__ import absolute_import

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from reid.lib.normalize import Normalize


__all__ = ['CbResNet', 'cbresnet18', 'cbresnet34', 'cbresnet50', 'cbresnet101',
           'cbresnet152']


class CbResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, num_triplet_features=0, n_splits=10, batch_size=128, adjustment=None):
        super(CbResNet, self).__init__()
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
        if depth not in CbResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = CbResNet.__factory[depth](pretrained=pretrained)

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
                self.feat_wise_classifiers = nn.ModuleList()
                for _ in range(self.n_splits):
                    fc = nn.Linear(self.split_conv_out_channels, self.num_classes)
                    init.normal_(fc.weight, std=0.001)
                    init.constant_(fc.bias, 0)
                    self.feat_wise_classifiers.append(fc)

                # Combined Adjustment
                self.combined_classifiers = nn.ModuleList()
                for i in range(self.n_splits):
                    # self.combined_bns.append(nn.BatchNorm1d(self.split_conv_out_channels))
                    fc = nn.Linear(self.split_conv_out_channels * 2, self.num_classes)
                    init.normal_(fc.weight, std=0.001)
                    init.constant_(fc.bias, 0)
                    self.combined_classifiers.append(fc)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, output_feature=None, mean_feats=None):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            else:
                x = module(x)

        if self.cut_at_pooling:
            return x

        if output_feature == 'Combined' or output_feature == 'mean_feat':
            split_h = x.size(2) / self.n_splits
            # splits_precs[n_splits, 128, 702]
            splits_feats = []
            splits_precs = []
            for i in range(self.n_splits):
                split_feat = F.avg_pool2d(
                    x[:, :, int(i * split_h): int((i + 1) * split_h), :],
                    (int((i + 1) * split_h) - int(i * split_h), x.size(-1)))

                split_feat = self.split_conv_list[i](split_feat)
                split_feat = split_feat.view(split_feat.size(0), -1)
                splits_feats.append(split_feat)

                # Combined Adjustment
                if output_feature == 'Combined':
                    cwa = torch.zeros(split_feat.size()).cuda()
                    if mean_feats != None:
                        mean_feat = self.drop(mean_feats[i])
                        pre_class = F.softmax(self.feat_wise_classifiers[i](split_feat), dim=1)
                        cwa = pre_class.mm(mean_feat)
                    prec = self.combined_classifiers[i](torch.cat([split_feat, cwa], dim=1))
                    splits_precs.append(prec)

            if output_feature == 'mean_feat':
                return torch.stack(splits_feats)
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


def cbresnet18(**kwargs):
    return CbResNet(18, **kwargs)


def cbresnet34(**kwargs):
    return CbResNet(34, **kwargs)


def cbresnet50(**kwargs):
    return CbResNet(50, **kwargs)


def cbresnet101(**kwargs):
    return CbResNet(101, **kwargs)


def cbresnet152(**kwargs):
    return CbResNet(152, **kwargs)