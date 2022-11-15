import collections
from typing import Callable, Optional, Tuple, OrderedDict

import numpy as np
import pandas as pd
import sklearn.metrics

from . import types, enums
from .uncertainty_metrics import (BrierScore, ExpectedCalibrationError,
                                  ClassificationR2, MaximumCalibrationError, AUROC)
from .uncertainty_metrics import (RegressionKendall,
                                  AbsoluteMiscalibrationArea)
from .uncertainty_metrics import ReliabilityDiagram, CVPPDiagram


def inference_evaluate(y_true: np.ndarray, y_pred: np.ndarray, task: enums.TaskType,
                       extra: Optional[types.EvalDict] = None, prefix: str = '', 
                       ci: bool = True) -> types.EvalDict:
    extra = extra or {}
    stats = collections.OrderedDict()
    if task == enums.TaskType.regression:
        metrics = {
            'R^2': sklearn.metrics.r2_score,
            # 'kendall': lambda x, y: scipy.stats.kendalltau(x, y)[0]
        }
    elif task == enums.TaskType.binary:
        metrics = {
            'accuracy': sklearn.metrics.accuracy_score
            # 'f1': sklearn.metrics.f1_score,
        }
    else:
        raise ValueError(f'{task} not implemented!')

    if ci:
        for key, metric_fn in metrics.items():
            metric_mean, metric_ci_bot, metric_ci_top = get_confidence_interval(y_true, y_pred, metric_fn)
            stats[f'{prefix}{key}'] = metric_mean
            stats[f'{prefix}{key}_CI_bot'] = np.abs(metric_ci_bot - metric_mean)
            stats[f'{prefix}{key}_CI_top'] = np.abs(metric_ci_top - metric_mean)
    else:
        for key, metric_fn in metrics.items():
            stats[f'{prefix}{key}'] = metric_fn(y_true, y_pred)

    stats.update(extra)
    return stats


def calibration_evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_err: np.array,
                         task: enums.TaskType, extra: Optional[types.EvalDict] = None, 
                         prefix: str = '', ci: bool = True):
    extra = extra or {}
    stats = collections.OrderedDict()
    if task == enums.TaskType.regression:
        scorers = {
            'absCVPParea': AbsoluteMiscalibrationArea(),
            # 'maxCVPPmiscalibration': MaximumMiscalibration(), 
            'kendall': RegressionKendall(),
        }

    elif task == enums.TaskType.binary:
        scorers = {
            'ece': ExpectedCalibrationError(),
            'mce': MaximumCalibrationError(),
            'auroc': AUROC(),
            'R^2': ClassificationR2(),
            'brier': BrierScore(),
        }
    else:
        raise ValueError(f'{task} not implemented!')

    if ci:
        for key, scorer in scorers.items():
            metric_mean, metric_ci_bot, metric_ci_top = get_calibration_confidence_interval(y_true, y_pred, y_err,
                                                                                            scorer.compute)
            stats[f'{prefix}{key}'] = metric_mean
            stats[f'{prefix}{key}_CI_bot'] = np.abs(metric_ci_bot - metric_mean)
            stats[f'{prefix}{key}_CI_top'] = np.abs(metric_ci_top - metric_mean)
    else:
        for key, scorer in scorers.items():
            stats[f'{prefix}{key}'] = scorer.compute(y_true, y_pred, y_err)
            
    stats.update(extra)
    return stats


def get_calibration_diagram_dataframe(y_true: np.ndarray, y_pred: np.ndarray, y_err: np.array,
                                      task: enums.TaskType, info: OrderedDict,
                                      num_bins: int = 10, n_samples: int = 1000):
    ''' Return dataframe with bootstrapped calibration plots.
    Dataframe can be plot with seaborn (long form). See gpmol.vis.py

    num_bins: specifies values q used to calculate calibration score
    n_samples: number of bootstrapping samples, if n_samples=1 no bootstrapping
    '''
    assert n_samples > 0, 'Number of bootstrap samples must be positive.'

    results = {
        'C(q)': [],
        'q': [],
        'feature': [],
        'method': []
    }

    n = len(y_true)
    qs = np.linspace(0, 1, num_bins)
    if task == enums.TaskType.regression:
        scorer = CVPPDiagram()
    elif task == enums.TaskType.binary:
        scorer = ReliabilityDiagram()
    else:
        raise ValueError(f'{task} not implemented!')

    for i in range(n_samples):
        idx = np.random.choice(n, int(n * 1.0), replace=True) if n_samples > 1 else np.arange(n)
        q, Cq = scorer.compute(y_true[idx], y_pred[idx], y_err[idx], num_bins=num_bins)
        results['q'].extend(q)
        results['C(q)'].extend(Cq)
        results['feature'].extend([info['feature']] * len(q))
        results['method'].extend([info['method']] * len(q))

    df = pd.DataFrame(results)

    return df


def get_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray, metric_fn: Callable,
                            n_samples: int = 1000) -> Tuple[float, float]:
    '''Bootstrap for 95% confidence interval.'''
    if n_samples == 1:
        return metric_fn(y_true, y_pred), 0

    results = np.zeros((n_samples, 1))
    n = len(y_true)
    for i in range(n_samples):
        idx = np.random.choice(n, int(n * 1.0), replace=True)
        results[i] = (metric_fn(y_true[idx], y_pred[idx]))

    m = np.mean(results)
    ci_top = np.percentile(results, 97.5)
    ci_bot = np.percentile(results, 2.5)

    return m, ci_bot, ci_top  # np.std(results)


def get_calibration_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray, y_err: np.array,
                                        metric_fn: Callable, n_samples: int = 1000) -> Tuple[float, float]:
    '''Bootstrap for 95% confidence interval.
    '''
    if n_samples == 1:
        return metric_fn(y_true, y_pred, y_err), 0

    results = np.zeros((n_samples, 1))
    n = len(y_true)
    for i in range(n_samples):
        idx = np.random.choice(n, int(n * 1.0), replace=True)
        results[i] = (metric_fn(y_true[idx], y_pred[idx], y_err[idx]))

    m = np.mean(results)
    ci_top = np.percentile(results, 97.5)
    ci_bot = np.percentile(results, 2.5)

    return m, ci_bot, ci_top
