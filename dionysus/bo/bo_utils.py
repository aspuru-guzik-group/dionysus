import os, sys
from copy import deepcopy
import numpy as np
import pandas as pd

from scipy.special import ndtr
from scipy.stats import norm

from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, FunctionTransformer

from .. import vis, files, enums, tasks, datasets


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
        mu = MinMaxScaler().fit_transform(mu.reshape(-1, 1)).squeeze()
        sigma = MinMaxScaler().fit_transform(sigma.reshape(-1, 1)).squeeze()

        return mu, sigma


    def __call__(self, mu, sigma, best_merit):
        ''' evaluate the acquisition function
        '''

        if self.task_type == 'regression':
            mu_norm, sigma_norm = self._normalize(mu, sigma)
        elif self.task_type == 'binary':
            mu_norm, sigma_norm = mu, sigma

        return self.func(mu_norm, sigma_norm, best_merit)


class IdentityScaler():

    def __init__(self, type='identity'):
        self.type = type

    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


def get_scalers(feature_set: enums.FeatureType, model: enums.Models, task: enums.TaskType):
    ''' scale the data in a dataset/model/feature specific manner
    if we are dealing with a classification task, we do not apply 
    any scaling to the targets

    Args:
        info (str): info provided as string with "dataset-model-features"

    '''

    feature_scaler, target_scaler = datasets.scaling_options(feature_set, model, task)

    def scaler_object(scaler_option):
        if scaler_option == None:
            scaler = IdentityScaler()
        elif scaler_option == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_option == 'standard':
            scaler = StandardScaler()
        elif scaler_option == 'quantile':
            scaler = QuantileTransformer()
        else:
            raise NotImplementedError
        return scaler

    return scaler_object(feature_scaler), scaler_object(target_scaler)



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
    p_metric = tasks.inference_evaluate(
        test_y[:, 0], y_pred[:, 0], config.tasks[0], {} 
    )
    c_metric = tasks.calibration_evaluate(
        test_y[:, 0], y_pred[:, 0], y_err[:, 0], config.tasks[0], {} 
    )

    p_metric.update(c_metric)

    return pd.DataFrame(p_metric, index=[0]) 




