from __future__ import absolute_import

from .resnet import *
from .ft_resnet import *
from .cl_resnet import *
from .cb_resnet import *

__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'clresnet18': clresnet18,
    'clresnet34': clresnet34,
    'clresnet50': clresnet50,
    'clresnet101': clresnet101,
    'clresnet152': clresnet152,
    'ftresnet18': ftresnet18,
    'ftresnet34': ftresnet34,
    'ftresnet50': ftresnet50,
    'ftresnet101': ftresnet101,
    'ftresnet152': ftresnet152,
    'cbresnet18': cbresnet18,
    'cbresnet34': cbresnet34,
    'cbresnet50': cbresnet50,
    'cbresnet101': cbresnet101,
    'cbresnet152': cbresnet152,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
