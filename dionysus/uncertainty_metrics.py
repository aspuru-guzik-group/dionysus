#!/usr/bin/env python

from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import simps
from scipy.stats import pearsonr, spearmanr, norm, kendalltau
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, r2_score


class AbstractRegressionMetric(ABC):
    ''' This class implements uncertainty metrics for regression
    tasks

    required methods:
        - compute
    '''

    @abstractmethod
    def compute(self, y_true, y_pred, y_err, **kwargs):
        ''' compute the metric

        Args:
            y_true (np.ndarray): array of true targets , shape (# samples, # targets)
            y_pred (np.ndarray): array of predictive means, shape (# samples, # targets)
            y_err (np.ndarray): array of predictive stds, shape (# samples, # targets)

        Returns:
            res (np.ndarray): the metric value
        '''


class AbstractClassificationMetric(ABC):
    ''' This class implements uncertainty metrics for binary classification
    tasks

    required methods:
        - compute
    '''

    @abstractmethod
    def compute(self, y_true, y_pred, y_prob, **kwargs):
        ''' compute the metric

        Args:
            pred_y (np.ndarray): array of predicted classes, shape (# samples, # classes)
            pred_mu (np.ndarray): array of predictive probabilities, shape (# samples, # classes)
            true (np.ndarray): array of true labels, shape (# samples, # classes, # samples)

        Returns:
            res (np.ndarray): the metric value
        '''


# Regression -------------------------------------------------------------------

class RegressionSpearman(AbstractRegressionMetric):
    ''' Spearman's correlation coefficient between the models absolute error
    and the uncertainty - non-linear correlation
    '''

    def __init__(self, name='spearman'):
        self.name = name

    def compute(self, y_true, y_pred, y_err):
        ae = np.abs(y_true - y_pred)
        res, _ = spearmanr(ae.ravel(), y_err.ravel())
        return res


class RegressionPearson(AbstractRegressionMetric):
    ''' Pearson's correlation coefficient between the models absolute error
    and the uncertainty - linear correlation
    '''

    def __init__(self, name='pearson'):
        self.name = name

    def compute(self, y_true, y_pred, y_err):
        ae = np.abs(y_true - y_pred)
        res, _ = pearsonr(ae.ravel(), y_err.ravel())
        return res


class RegressionKendall(AbstractRegressionMetric):
    ''' Kendall's correlation coefficient between the models absolute error
    and the uncertainty - linear correlation
    '''

    def __init__(self, name='kendall'):
        self.name = name

    def compute(self, y_true, y_pred, y_err):
        ae = np.abs(y_true - y_pred)
        res, _ = kendalltau(ae.ravel(), y_err.ravel())
        return res


class CVPPDiagram(AbstractRegressionMetric):
    ''' Metric introduced in arXiv:2010.01118 [cs.LG] based on cross-validatory
    predictive p-values
    '''

    def __init__(self, name='cvpp'):
        self.name = name

    @staticmethod
    def c(y_true, y_pred, y_err, q):
        lhs = np.abs((y_pred - y_true) / y_err)
        rhs = norm.ppf(((1.0 + q) / 2.0), loc=0., scale=1.)
        return np.sum((lhs < rhs).astype(int)) / y_true.shape[0]

    def compute(self, y_true, y_pred, y_err, num_bins=10):
        qs = np.linspace(0, 1, num_bins)
        Cqs = np.empty(qs.shape)
        for ix, q in enumerate(qs):
            Cqs[ix] = self.c(y_true, y_pred, y_err, q)

        return qs, Cqs


class MaximumMiscalibration(AbstractRegressionMetric):
    ''' Miscalibration area metric with CVPP
    WARNING - this metric only diagnoses systematic over- or under-
    confidence, i.e. a model that is overconfident for ~half of the
    quantiles and under-confident for ~half will still have a MiscalibrationArea
    of ~0.
    '''

    def __init__(self, name='mmc'):
        self.name = name

    def compute(self, y_true, y_pred, y_err, num_bins=10):
        scorer = CVPPDiagram()
        qs, Cqs = scorer.compute(y_true, y_pred, y_err, num_bins=num_bins)

        # compute area
        res = np.max(np.abs(Cqs - qs))
        return res


class MiscalibrationArea(AbstractRegressionMetric):
    ''' Miscalibration area metric with CVPP
    WARNING - this metric only diagnoses systematic over- or under-
    confidence, i.e. a model that is overconfident for ~half of the
    quantiles and under-confident for ~half will still have a MiscalibrationArea
    of ~0.
    '''

    def __init__(self, name='ma'):
        self.name = name

    def compute(self, y_true, y_pred, y_err, num_bins=10):
        scorer = CVPPDiagram()
        qs, Cqs = scorer.compute(y_true, y_pred, y_err, num_bins=num_bins)

        # compute area
        res = simps(Cqs - qs, qs)
        return res


class AbsoluteMiscalibrationArea(AbstractRegressionMetric):
    ''' absolute miscalibration area metric with CVPP
    '''

    def __init__(self, name='ama'):
        self.name = name

    def compute(self, y_true, y_pred, y_err, num_bins=10):
        scorer = CVPPDiagram()
        qs, Cqs = scorer.compute(y_true, y_pred, y_err, num_bins=num_bins)

        # compute area
        res = simps(np.abs(Cqs - qs), qs)
        return res


class NLL(AbstractRegressionMetric):
    ''' Negative log-likelihood
    '''

    def __init__(self, name='nll'):
        self.name = name

    def compute(self, y_true, y_pred, y_err, **kwargs):
        res = 1. / (2. * y_true.shape[0]) + y_true.shape[0] * np.log(2 * np.pi) \
              + np.sum(np.log(y_pred)) + np.sum(np.square(y_pred - y_true) / y_err)
        return res


class CalibratedNLL(AbstractRegressionMetric):
    ''' calibrated negative log-likelihood - calibrate the uncertainty
    so that it more closely resembles variances, this assumes the two are
    linearly related. This can be used for UQ methods whose uncertainty estimates
    are not intended to be used as variances (e.g. distances-)

    i.e. sigma^2(x) = a*U(x) + b
    '''

    def __init__(self, name='cnll'):
        self.name = name

    def compute(self, y_true, y_pred, y_err, **kwargs):
        # TODO : implement this
        return None


# Classification ---------------------------------------------------------------


class BrierScore(AbstractClassificationMetric):
    ''' Brier score metric
    WARNING: This formulation is only compatible with binary classification
    tasks, i.e. toxic (1) or non-toxic (0)
    '''

    def __init__(self, name='brier'):
        self.name = name

    def compute(self, y_true, y_pred, y_prob):
        res = brier_score_loss(y_true.ravel(), y_prob.ravel())
        return res


class ReliabilityDiagram(AbstractClassificationMetric):
    ''' compute the reliability diagram (https://arxiv.org/abs/1706.04599,
    https://arxiv.org/abs/1807.00263)
    WARNING: This metric is NOT scalar valued and is meant for plotting
    mean_conf to be plotted on x-axis, acc_tab to be plotted on y-axis as per
    Guo et al.
    '''

    def __init__(self, name='reldia'):
        self.name = name

    def compute(self, y_true, y_pred, y_prob, num_bins=10):
        accuracy, confidence = calibration_curve(y_true, y_prob, normalize=True, n_bins=num_bins, strategy='uniform')
        return confidence, accuracy


class ExpectedCalibrationError(AbstractClassificationMetric):
    ''' compute the expeced calibration error (https://arxiv.org/abs/1706.04599,
    https://arxiv.org/abs/1807.00263)
    '''

    def __init__(self, name='ece'):
        self.name = name

    def compute(self, y_true, y_pred, y_prob, num_bins=10):
        scorer = ReliabilityDiagram()
        confidence, accuracy = scorer.compute(y_true, y_pred, y_prob, num_bins=num_bins)
        res = np.mean(np.abs(confidence - accuracy))
        return res


class MaximumCalibrationError(AbstractClassificationMetric):
    ''' compute the maximum calibration error calibration (https://arxiv.org/abs/1706.04599,
    https://arxiv.org/abs/1807.00263)
    '''

    def __init__(self, name='mce'):
        self.name = name

    def compute(self, y_true, y_pred, y_prob, num_bins=10):
        scorer = ReliabilityDiagram()
        confidence, accuracy = scorer.compute(y_true, y_pred, y_prob, num_bins=num_bins)
        res = np.max(np.absolute(confidence - accuracy))
        return res


class AP(AbstractClassificationMetric):

    def __init__(self, name='ap'):
        self.name = name

    def compute(self, y_true, y_pred, y_prob):
        # flatten for binary classifications
        if y_true.shape[-1] == 1:
            y_true = np.squeeze(y_true)
            y_prob = np.squeeze(y_prob)
        score = average_precision_score(y_true, y_prob)
        return score


class AUROC(AbstractClassificationMetric):

    def __init__(self, name='auroc'):
        self.name = name

    def compute(self, y_true, y_pred, y_prob):
        # flatten for binary classifications
        if y_true.shape[-1] == 1:
            y_true = np.squeeze(y_true)
            y_prob = np.squeeze(y_prob)
        score = roc_auc_score(y_true, y_prob)
        return score


class ClassificationSpearman(AbstractClassificationMetric):
    ''' Spearman's correlation coefficient between the models absolute error
    and the uncertainty - non-linear correlation
    '''

    def __init__(self, name='spearman'):
        self.name = name

    def compute(self, y_true, y_pred, y_prob):
        scorer = ReliabilityDiagram()
        confidence, accuracy = scorer.compute(y_true, y_pred, y_prob)
        res, _ = pearsonr(accuracy.ravel(), confidence.ravel())
        return res


class ClassificationPearson(AbstractClassificationMetric):
    ''' Spearman's correlation coefficient between the models absolute error
    and the uncertainty - non-linear correlation
    '''

    def __init__(self, name='pearson'):
        self.name = name

    def compute(self, y_true, y_pred, y_prob):
        scorer = ReliabilityDiagram()
        confidence, accuracy = scorer.compute(y_true, y_pred, y_prob)
        res, _ = spearmanr(accuracy.ravel(), confidence.ravel())
        return res


class ClassificationKendall(AbstractClassificationMetric):
    ''' Spearman's correlation coefficient between the models absolute error
    and the uncertainty - non-linear correlation
    '''

    def __init__(self, name='kendall'):
        self.name = name

    def compute(self, y_true, y_pred, y_prob):
        scorer = ReliabilityDiagram()
        confidence, accuracy = scorer.compute(y_true, y_pred, y_prob)
        res, _ = kendalltau(accuracy.ravel(), confidence.ravel())
        return res


class ClassificationR2(AbstractClassificationMetric):
    ''' Spearman's correlation coefficient between the models absolute error
    and the uncertainty - non-linear correlation
    '''

    def __init__(self, name='r2'):
        self.name = name

    def compute(self, y_true, y_pred, y_prob):
        scorer = ReliabilityDiagram()
        confidence, accuracy = scorer.compute(y_true, y_pred, y_prob)
        res = r2_score(accuracy.ravel(), confidence.ravel())
        return res


# DEBUGGING
if __name__ == '__main__':
    pass
