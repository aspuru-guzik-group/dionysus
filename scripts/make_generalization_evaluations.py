import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT_DIR = '..'
sys.path.append(ROOT_DIR)

import dionysus
import numpy as np
import pandas as pd
from absl import app, flags
import multiprocessing
from functools import partial

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

from dionysus import vis, files, enums

FEATURE_ORDER = enums.VECTOR_FEATURES + enums.GRAPH_FEATURES
METHOD_ORDER = list(enums.Models)
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', None, 'Dataset to process')
flags.DEFINE_string('split_type', 'cluster', 'Types: cluster, incremental-diverse, incrememntal-standard. Default cluster.')
flags.DEFINE_integer('task_index', -1, 'Select task for multi-target dataset. Set -1 to use all.')


def main(_):
    vis.plotting_settings()
    sns.set_context('talk', font_scale=2.0)
    mpl.rcParams['axes.linewidth'] = 2.5
    mpl.rcParams['lines.markersize'] = 15.0
    mpl.rcParams['legend.markerscale'] = 2.0
    mpl.rcParams['legend.fontsize'] = 'small'
    mpl.rcParams['figure.figsize'] = [40.0, 10.0]
    DATASET, TASK_INDEX = FLAGS.dataset, FLAGS.task_index

    base_dir = os.path.join(dionysus.files.get_data_dir(ROOT_DIR), DATASET)
    _, config, _ = dionysus.datasets.load_dataset(DATASET, base_dir)
    config.task = config.tasks[TASK_INDEX]
    work_dir = os.path.join(base_dir, 'generalization')
    res_dir = os.path.join(work_dir, 'results')
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)

    def get_filename(ftype, fname_args=None):
        return dionysus.files.get_filename(work_dir, ftype, fname_args)

    def get_statistics_from_prediction(fname):
        # print(f'Reading {i} of {len(pred_files)} files...')
        info = files.get_file_info(fname)
        split_info = fname.split('-')
        if len(split_info) > 2:
            split_type = ''.join([split_info[1], '-', split_info[-1].split('_')[0]])
        else:    
            split_type = split_info[-1].split('_')[0]
        smi_split, _, _ = dionysus.datasets.load_task(DATASET, enums.FeatureType.mfp, enums.Models.sngp, 
                                                base_dir, result_dir=work_dir, task_index=TASK_INDEX,
                                                split_type=split_type, verbose=False)

        # update info on cluster splits
        train_size = (len(smi_split.train) + len(smi_split.val))  / (len(smi_split.values) - len(smi_split.test))
        method = info['method'].split('-')[0]
        split = info['method'].split('-')[1]
        info.update({'method': method, 'cluster': split, 'split': 'test', 'train_size': train_size})

        # map predictions to smiles
        fn_true = dionysus.utils.SmilesMap(fname, values='y_true')
        fn_pred = dionysus.utils.SmilesMap(fname, values='y_pred')
        fn_err = dionysus.utils.SmilesMap(fname, values='y_err')

        # get test results
        y_true = fn_true(smi_split.test)
        y_pred = fn_pred(smi_split.test)
        y_err = fn_err(smi_split.test)

        # loop through multilabels
        p_metric = []
        c_metric = []
        for i in range(y_true.shape[-1]):
            # get prediction and uncertainty results
            p_metric.append(dionysus.tasks.inference_evaluate(y_true[:, i], y_pred[:, i], config.task, {}, ci=False))
            c_metric.append(dionysus.tasks.calibration_evaluate(y_true[:, i], y_pred[:, i], y_err[:, i], config.task, {}, ci=False))

        # average over multiple outputs
        p_metric = pd.DataFrame(p_metric).mean().to_dict()
        p_metric.update(info)
        c_metric = pd.DataFrame(c_metric).mean().to_dict()
        c_metric.update(info)

        return p_metric, c_metric

    assert FLAGS.split_type in ['cluster', 'incremental-diverse', 'incremental-standard'], 'Invalid cluster type.'
    fname = get_filename('predictions', ['*', '*'])

    pred_files = dionysus.utils.get_matching_files(fname)
    pred_files = [p for p in pred_files if FLAGS.split_type in p]
    
    # process in parallel
    output = dionysus.utils.parallel_map(get_statistics_from_prediction, pred_files)

    # dataframes of prediction and calibration scores with confidence intervals
    pred_results, calib_results = zip(*output)
    pred_results, calib_results = list(pred_results), list(calib_results)
    pred_results = pd.DataFrame(pred_results)
    calib_results = pd.DataFrame(calib_results)

    # remove graphembeddings (they are unfairly trained on whole dataset)
    pred_results = pred_results[pred_results['feature'] != 'graphembed']
    calib_results = calib_results[calib_results['feature'] != 'graphembed']

    # rename statistics of interest
    if config.task == enums.TaskType.regression:
        p_stat, c_stat = '$R^2$', 'Absolute Miscalibration Area'
        pred_results = pred_results.rename(columns={'R^2': p_stat, 'train_size': 'Training set (%)'})
        calib_results = calib_results.rename(columns={
            'absCVPParea': c_stat, 
            'train_size': 'Training set (%)',
            # 'kendall': 'Kendall tau'
        })
    elif config.task == enums.TaskType.binary:
        p_stat, c_stat = 'AUROC', 'Expected Calibration Error'
        pred_results = calib_results.rename(columns={'auroc': p_stat, 'train_size': 'Training set (%)'})
        calib_results = calib_results.rename(columns={
            'ece': c_stat, 
            'train_size': 'Training set (%)',
            # 'kendall': 'Kendall tau'
        })
    else:
        raise NotImplemented('Not implemented for task type.')

    pred_results['method'] = pred_results['method'].astype('category').cat.set_categories(METHOD_ORDER).cat.remove_unused_categories()
    calib_results['method'] = calib_results['method'].astype('category').cat.set_categories(METHOD_ORDER).cat.remove_unused_categories()

    # plot prediction splits
    results = {'mean': [], 'median': [], 'kendall': [], 'method': [], 'feature': [], 'statistic': []}
    fig, ax = plt.subplots(1, len(pred_results['method'].unique()), sharex=True, sharey=True)
    ax = ax.flatten()
    handles, labels = [], []
    for a, (meth, grp_df) in zip(ax, pred_results.groupby('method')):
        grp_df['feature'] = grp_df['feature'].astype('category').cat.set_categories(FEATURE_ORDER).cat.remove_unused_categories()
        a.set_xlim([0,1])
        if config.task == enums.TaskType.regression:
            a.set_ylim([-0.1,1])
        elif config.task == enums.TaskType.binary:
            a.set_ylim([0.4,1])
        else:
            raise NotImplemented('Not implemented for task type.')

        g = sns.scatterplot(data=grp_df, y=p_stat, x='Training set (%)', hue='feature', ax=a, legend='brief', palette=vis.CMAP)
        g.get_legend().set_title(None)
        a.legend(bbox_to_anchor=(0.0, 1.07), loc='lower left')
        a.set_title(meth)
        a.set_box_aspect(1)

        # calculate statistics and save
        for feat, ggrp_df in grp_df.groupby('feature'):
            results['kendall'].append(kendalltau(ggrp_df['Training set (%)'].to_numpy(), ggrp_df[p_stat].to_numpy())[0])
            results['mean'].append(np.mean(ggrp_df[p_stat]))
            results['median'].append(np.median(ggrp_df[p_stat]))
            results['method'].append(meth)
            results['feature'].append(feat)
            results['statistic'].append(p_stat)
    
    plt.savefig(os.path.join(res_dir, f'results_{FLAGS.split_type}_prediction.jpg'), bbox_inches='tight')
    pd.DataFrame(results).to_csv(os.path.join(res_dir, f'results_{FLAGS.split_type}_prediction.csv'), index=False)


    # plot calibration on splits
    results = {'mean': [], 'median': [], 'kendall': [], 'method': [], 'feature': [], 'statistic': []}
    fig, ax = plt.subplots(1, len(calib_results['method'].unique()), sharex=True, sharey=True)
    ax = ax.flatten()
    for a, (meth, grp_df) in zip(ax, calib_results.groupby('method')):
        grp_df['feature'] = grp_df['feature'].astype('category').cat.set_categories(FEATURE_ORDER).cat.remove_unused_categories()
        g = sns.scatterplot(data=grp_df, y=c_stat, x='Training set (%)', hue='feature', ax=a, palette=vis.CMAP)
        g.get_legend().set_title(None)
        a.legend(bbox_to_anchor=(0.0, 1.07), loc='lower left')
        a.set_xlim([0,1])
        a.set_ylim([0, 0.5])
        a.set_title(meth)
        a.set_box_aspect(1)

        # calculate statistics and save
        for feat, ggrp_df in grp_df.groupby('feature'):
            # for stat in ['Kendall tau', c_stat]:
            stat = c_stat
            results['kendall'].append(kendalltau(ggrp_df['Training set (%)'].to_numpy(), ggrp_df[stat].to_numpy())[0])
            results['mean'].append(np.mean(ggrp_df[stat]))
            results['median'].append(np.median(ggrp_df[stat]))
            results['statistic'].append(stat)
            results['method'].append(meth)
            results['feature'].append(feat)

    plt.savefig(os.path.join(res_dir ,f'results_{FLAGS.split_type}_calibration.jpg'), bbox_inches='tight')
    pd.DataFrame(results).to_csv(os.path.join(res_dir, f'results_{FLAGS.split_type}_calibration.csv'), index=False)


if __name__ == '__main__':
    flags.mark_flag_as_required('dataset')
    app.run(main)
