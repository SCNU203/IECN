from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from reid.lib.normalize import Normalize


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, num_triplet_features=0, n_splits=10, batch_size=128, adjustment=None):
        super(ResNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.n_splits = n_splits
        self.num_splits = num_classes // self.n_splits
        self.batch_size = batch_size
        self.adjustment = adjustment
        if self.num_splits * self.n_splits != num_features:
            self.num_splits += 1

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

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
            if self.dropout >= 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)

                # feature-wise Adjustment
                if self.adjustment == 'feature-wise':
                    self.feat_wise_classifiers = nn.ModuleList()
                    sum = 0
                    for i in range(self.n_splits):
                        sum += self.num_splits
                        if sum <= self.num_features:
                            fc = nn.Linear(self.num_splits, self.num_classes)
                        else:
                            sum = self.num_features - sum + self.num_splits
                            fc = nn.Linear(self.num_features + sum, self.num_classes)
                        init.normal_(fc.weight, std=0.001)
                        init.constant_(fc.bias, 0)
                        self.feat_wise_classifiers.append(fc)

                # class-wise Adjustment
                if self.adjustment == 'class-wise':
                    self.class_wise_classifier = nn.Linear(self.num_features * 2, self.num_classes)
                    init.normal_(self.class_wise_classifier.weight, std=0.001)
                    init.constant_(self.class_wise_classifier.bias, 0)

                # Combined Adjustment
                if self.adjustment == 'Combined':
                    self.combined_classifiers = nn.ModuleList()
                    sum = 0
                    for i in range(self.n_splits):
                        sum += self.num_splits
                        if sum <= self.num_features:
                            fc = nn.Linear(self.num_splits * 2, self.num_classes)
                        else:
                            sum = self.num_features - sum + self.num_splits
                            fc = nn.Linear(self.num_features + sum * 2, self.num_classes)
                        init.normal_(fc.weight, std=0.001)
                        init.constant_(fc.bias, 0)
                        self.combined_classifiers.append(fc)


        if not self.pretrained:
            self.reset_params()

    def forward(self, x, output_feature=None, mean_feat=None):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            else:
                x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

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
        if output_feature == 'mean_feat':
            return x

        # class-wise Adjustment
        if output_feature == 'class-wise':
            # pre_class[128,702]
            pre_class = self.classifier(x)
            # cwa[128,4096]
            cwa = pre_class.mm(mean_feat) / self.num_classes
            output = self.class_wise_classifier(torch.cat([x, cwa], dim=1))
            # output[128,702]
            return output

        # feature-wise Adjustment
        if output_feature == 'feature-wise':
            # splits_precs[n_splits, 128, 702]
            splits_precs = []
            for i in range(self.n_splits):
                prec = self.feat_wise_classifiers[i](x[:, self.num_splits * i : self.num_splits * (i + 1)])
                splits_precs.append(prec)
            # output[n_splits,128,702]
            return torch.stack(splits_precs)

        # Combined Adjustment
        if output_feature == 'Combined':
            # pre_class[128,702]
            pre_class = self.classifier(x)
            # cwa[128,4096]
            cwa = pre_class.mm(mean_feat) / self.num_classes
            # splits_precs[n_splits, 128, 702]
            splits_precs = []
            for i in range(self.n_splits):
                prec = self.combined_classifiers[i](torch.cat([x[:, self.num_splits * i: self.num_splits * (i + 1)], cwa[:, self.num_splits * i: self.num_splits * (i + 1)]], dim=1))
                splits_precs.append(prec)
            # output[n_splits,128,702]
            return torch.stack(splits_precs)

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


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
