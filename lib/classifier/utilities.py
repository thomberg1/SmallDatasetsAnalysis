# coding: utf-8

import math
import numpy as np
from collections import Counter


import torch
import torch.nn.init as init
from torch import nn
from torch.autograd import Variable

#####################################################################################################################

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

#####################################################################################################################

def mini_batch(X, Y, n, shuffle=False):
    X_iterable, Y_iterable = X, Y
    l = len(X)
    for ndx in range(0, l, n):
        if shuffle:
            idx_data = np.random.permutation(min(ndx + n, l) - ndx)
            yield X_iterable[ndx:min(ndx + n, l)][idx_data], Y_iterable[ndx:min(ndx + n, l)][idx_data]
        else:
            yield X_iterable[ndx:min(ndx + n, l)], Y_iterable[ndx:min(ndx + n, l)]

#####################################################################################################################

def exp_decay (epoch):
    return  0.5  * 1 / (1  + epoch * 0.5 )

def exp_decay1(epoch):
    lr = 0.2
    lr_g = 0.05
    lr_b = 0.75
    return  lr * (1  + epoch * lr_g )**-lr_b

def step_decay(epoch):
    lr = 0.2
    lr_drop = 0.78         # factor / percent to drop
    lr_epochs_drop = 16.0  # no of epochs after which to drop lr
    lr_min = 0.01
    return max( lr * math.pow(lr_drop, math.floor((1 + epoch) / lr_epochs_drop)), lr_min)


    # lambda epoch: max(math.pow(0.78, math.floor((1 + epoch) / 11.0)), 0.4)

#####################################################################################################################

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x, async=False):
    if torch.cuda.is_available():
        x = x.cuda(async=async)
    return Variable(x)

###################################################################################################################

from imblearn.base import *
from imblearn.utils import check_ratio, check_target_type, hash_X_y
import logging


class OutlierSampler(SamplerMixin):
    def __init__(self, threshold=1.5, memory=None, verbose=0):
        self.threshold = threshold
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

    def sample(self, X, y):
        # Check the consistency of X and y
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])

        check_is_fitted(self, 'X_hash_')
        self._check_X_y(X, y)

        X_out, y_out = self._sample(X, y)

        return X_out, y_out

    def _sample(self, X, y):
        outliers  = []
        for col in X.T: # loop over feature columns
            Q1 = np.percentile(col, 25)  # Calculate Q1 (25th percentile of the data) for the given feature
            Q3 = np.percentile(col, 75) # Calculate Q3 (75th percentile of the data) for the given feature

            step = self.threshold * (Q3 - Q1)  # Use the interquartile range to calculate an outlier step

            feature_outliers = np.where(~((col >= Q1 - step) & (col <= Q3 + step)))[0]
            outliers.extend(feature_outliers)

        # Find the data points that where considered outliers for more than one feature
        multi_feature_outliers = list((Counter(outliers) - Counter(set(outliers))).keys())

        X_out = np.delete(X, multi_feature_outliers, axis=0)
        y_out = np.delete(y, multi_feature_outliers, axis=0)

        if self.verbose:
            print('Sampled - reduced points form / to: ', X.shape, X_out.shape)
        return X_out, y_out

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        y = check_target_type(y)
        self.X_hash_, self.y_hash_ = hash_X_y(X, y)

        self._fit( X, y)

        return self

    def _fit(self, X, y):
        if self.verbose:
            print('OutlierSampler Fitted X/y: ', X.shape, y.shape)
        return self

    def fit_sample(self, X, y):
        return self.fit(X, y).sample(X, y)

#######################################################################################################################
# https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5

def torch_weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)

#######################################################################################################################