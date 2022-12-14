#!/usr/bin/env python

import os, sys
from copy import deepcopy
import numpy as np

from scipy.special import ndtr
from scipy.stats import norm

from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from absl import app, flags


# TODO: this is a stupid hack, we should make a software package instead
cwd = os.getcwd()
split = cwd.split('/')
loc = split.index('gpmol_v2')
num_dir = len(split)-loc-1
ROOT_DIR = '/'.join(['..' for _ in range(num_dir)]) + '/'
sys.path.append(ROOT_DIR)

import gpmol
from gpmol import vis, files, enums



class AcquisitionFunction():
    ''' acquisition functions, all are expected to be maximized
    '''
    def __init__(self, acq_func_type='ei',  task_type='regression', beta=0.2):
        self.acq_func_type = acq_func_type
        self.task_type = task_type
        self.beta = beta
        self.delta = 1.-beta
        print('BETA : ', self.beta)
        print('DELTA :', self.delta)
        self.func = getattr(self, self.acq_func_type)


    def ei(
        self,
        mu,
        sigma,
        best_merit,
        xi=0.01,
    ):
        ''' expected improvement acquistion for minimization
        '''
        Z = (best_merit - mu - xi) / sigma
        return (best_merit - mu - xi)*ndtr(Z) + sigma*norm.pdf(Z)

    def ucb(
        self,
        mu,
        sigma,
        best_merit,
    ):
        ''' upper confidence bound acquisition for minimization
         - higher values the better --> best merit not needed here
        '''
        return -(self.delta*mu - self.beta*sigma)

    def lcb(
        self,
        mu,
        sigma,
        best_merit,
        beta=0.2,
    ):
        ''' lower confidence bound acquisition for minimization
         - higher values the better --> best merit not needed here
        '''
        return -(self.delta*mu + self.beta*sigma)

    def variance_sampling(
        self, 
        mu, 
        sigma, 
        best_merit,
        ):
        ''' pure variance based sampling for active learning
        '''
        return sigma


    def greedy_binary(
        self,
        mu, 
        sigma, 
        best_merit,
        ):
        ''' sampling strategy used for searches of positive instances in
        a dataset of binary objectives (0 and 1 labels). Maximizing the 
        negative of this function is then equivalent to searching for 
        negative labels, i.e. 0s
        '''
        return -mu


    def _normalize(self, mu, sigma):
        ''' normalize (min-max scale) the mean and variance values
        '''
        # mu
        min_ = np.amin(mu, axis=0)
        max_ = np.amax(mu, axis=0)

        ixs = np.where(np.abs(max_ - min_) < 1e-10)[0]
        if not ixs.size == 0:
            max_[ixs] = np.ones_like(ixs)
            min_[ixs] = np.zeros_like(ixs)
        mu = (mu - min_) / (max_ - min_)

        # sigma
        min_ = np.amin(sigma, axis=0)
        max_ = np.amax(sigma, axis=0)

        ixs = np.where(np.abs(max_ - min_) < 1e-10)[0]
        if not ixs.size == 0:
            max_[ixs] = np.ones_like(ixs)
            min_[ixs] = np.zeros_like(ixs)
        sigma = (sigma - min_) / (max_ - min_)

        return mu, sigma


    def __call__(self, mu, sigma, best_merit):
        ''' evaluate the acquisition function
        '''

        if self.task_type == 'regression':
            mu_norm, sigma_norm = self._normalize(mu, sigma)
        elif self.task_type == 'binary':
            mu_norm, sigma_norm = mu, sigma

        return self.func(mu_norm, sigma_norm, best_merit)



def get_scalers(info, task_type):
    ''' scale the data in a dataset/model/feature specific manner
    if we are dealing with a classification task, we do not apply 
    any scaling to the targets

    Args:
        info (str): info provided as string with "dataset-model-features"

    '''

    scaling_types = {
        # ngboost
        'ngboost-mfp': {'X': 'identity', 'y': 'identity'},
        'ngboost-mordred': {'X': 'identity', 'y': 'identity'},
        'ngboost-graphembed': {'X': 'identity', 'y': 'identity'},
        # gp
        'gp-mfp': {'X': 'identity', 'y': 'standardization'},
        'gp-mordred': {'X': 'quantile', 'y': 'standardization'},
        'gp-graphembed': {'X': 'identity', 'y': 'standardization'},
        # bnn
        'bnn-mfp': {'X': 'identity', 'y': 'standardization'},
        'bnn-mordred': {'X': 'quantile', 'y': 'standardization'},
        'bnn-graphembed': {'X': 'identity', 'y': 'standardization'},
        # gnngp
        'gnngp-graphnet': {'X': 'identity', 'y': 'standardization'},
        # sngp
        # bnn
        'sngp-mfp': {'X': 'identity', 'y': 'standardization'},
        'sngp-mordred': {'X': 'quantile', 'y': 'standardization'},
        'sngp-graphembed': {'X': 'identity', 'y': 'standardization'},
    }

    if 'random-' in info:
        X_type = 'identity'
    else:
        X_type = scaling_types[info]['X']
    # if we have a binary task, we dont scale the targets
    # TODO: remove this, it is not nessecary
    if task_type ==  'binary':
        y_type = 'identity'
    elif task_type == 'regression':
        if 'random-' in info:
            y_type = 'identity'
        else:
            y_type = scaling_types[info]['y']



    if X_type == 'identity':
        X_scaler = IndentityScaler()
    elif X_type == 'minmax':
        X_scaler = MinMaxScaler()
    elif X_type == 'standardization':
        X_scaler = StandardScaler()
    elif X_type == 'quantile':
        X_scaler = QuantileTransformer()
    else:
        raise NotImplementedError

    if y_type == 'identity':
        y_scaler = IndentityScaler()
    elif y_type == 'minmax':
        y_scaler = MinMaxScaler()
    elif y_type == 'standardization':
        y_scaler = StandardScaler()
    elif y_type == 'quantile':
        y_scaler = QuantileTransformer()
    else:
        raise NotImplementedError

    return X_scaler, y_scaler


def get_model(info):
    ''' returns the gpmol model object to use
    '''
    return None


class IndentityScaler():

    def __init__(self, type='identity'):
        self.type = type

    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X



def predict_test_set(
    model,
    test_X, 
    test_y, 
    X_scaler,
    y_scaler,
    config,
):
    ''' make a prediction on the held-out test
    using the BO surrogate model and record the metrics
    '''

    # forward scale the features
    test_X_scal = X_scaler.transform(test_X)
    # make prediction with surrogate model
    y_pred_scal, y_err = model.predict_bo(test_X_scal)
    # reverse scale the predictions
    y_pred  = y_scaler.inverse_transform(y_pred_scal)

    # we will only consider scalar objectives here (i.e. the
    # first task index)
    p_metric = gpmol.tasks.inference_evaluate(
        test_y[:, 0], y_pred[:, 0], config.tasks[0], {} 
    )
    c_metric = gpmol.tasks.calibration_evaluate(
        test_y[:, 0], y_pred[:, 0], y_err[:, 0], config.tasks[0], {} 
    )

    return {'p_metric': p_metric, 'c_metric': c_metric}




