import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT_DIR = '..'
sys.path.append(ROOT_DIR)

import dionysus
import numpy as np
import pandas as pd
from absl import app, flags

from dionysus import vis, files, enums

FEATURE_ORDER = enums.VECTOR_FEATURES + enums.GRAPH_FEATURES
METHOD_ORDER = list(enums.Models)
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', None, 'Dataset to process')
flags.DEFINE_integer('task_index', -1, 'Select task for multi-target dataset. Set -1 to use all.')


def main(_):
    vis.plotting_settings()
    work_dir = os.path.join(dionysus.files.get_data_dir(ROOT_DIR), FLAGS.dataset)
    _, config, _ = dionysus.datasets.load_dataset(FLAGS.dataset, work_dir)
    config.task = config.tasks[FLAGS.task_index]

    smi_split, _, _ = dionysus.datasets.load_task(FLAGS.dataset, enums.FeatureType.mfp, enums.Models.sngp, work_dir,
                                               task_index=FLAGS.task_index)

    def get_filename(ftype, fname_args=None):
        return dionysus.files.get_filename(work_dir, ftype, fname_args)

    # search for all 
    pred_files = dionysus.utils.get_matching_files(get_filename('predictions', ['*', '*']))
    pred_results = []
    calib_results = []
    dia_results = []

    # go through files and create results in dataframes
    for fname in pred_files:
        info = files.get_file_info(fname)
        info.update({'split': 'test'})

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
            p_metric.append(dionysus.tasks.inference_evaluate(y_true[:, i], y_pred[:, i], config.task, {}))
            c_metric.append(dionysus.tasks.calibration_evaluate(y_true[:, i], y_pred[:, i], y_err[:, i], config.task, {}))

        p_metric = pd.DataFrame(p_metric).mean().to_dict()
        p_metric.update(info)
        c_metric = pd.DataFrame(c_metric).mean().to_dict()
        c_metric.update(info)

        pred_results.append(p_metric)
        calib_results.append(c_metric)

        if y_true.shape[-1] == 1:
            n_samples = 100 if config.task == enums.TaskType.regression else 1
            # only plot diagram for single task jobs
            df = dionysus.tasks.get_calibration_diagram_dataframe(y_true, y_pred, y_err, config.task, info,
                                                               n_samples=n_samples)
            dia_results.append(df)
        else:
            if FLAGS.task_index != -1:
                raise ValueError('Multilabel output found. Please set task_index = -1.')

    # dataframes of prediction and calibration scores with confidence intervals
    pred_results = pd.DataFrame(pred_results)
    calib_results = pd.DataFrame(calib_results)

    # add zeroes where graphs vs vector features are used
    null_results = {'method': [], 'feature': [], 'split': []}
    for m in np.unique(pred_results['method']):
        m_results = pred_results[pred_results['method'] == m]
        for feat in np.unique(pred_results['feature']):
            if feat not in np.unique(m_results['feature']):
                null_results['method'].append(m)
                null_results['feature'].append(feat)
                null_results['split'].append('test')
    null_results = pd.DataFrame(null_results)  # dataframe of non-existent method/feature pair
    pred_results = pd.concat([pred_results, null_results]).fillna(0)
    calib_results = pd.concat([calib_results, null_results]).fillna(0)

    # sort the features and models based on assigned lists
    pred_results['method'] = pred_results['method'].astype('category').cat.set_categories(METHOD_ORDER)
    pred_results['feature'] = pred_results['feature'].astype('category').cat.set_categories(FEATURE_ORDER)
    calib_results['method'] = calib_results['method'].astype('category').cat.set_categories(METHOD_ORDER)
    calib_results['feature'] = calib_results['feature'].astype('category').cat.set_categories(FEATURE_ORDER)
    pred_results = pred_results.sort_values(['method', 'feature'])
    calib_results = calib_results.sort_values(['method', 'feature'])

    # print results 
    pred_results.to_csv(os.path.join(work_dir, 'prediction_results.csv'), index=False)
    calib_results.to_csv(os.path.join(work_dir, 'calibration_results.csv'), index=False)

    # compare metrics on prediction for all models
    metrics = pred_results.columns[pred_results.columns.str.contains('CI')].str.split('_').str.get(0)
    for m in metrics:
        fig = vis.plot_metrics(pred_results, metric=m, compare_by='feature', group_by='method', task=config.task)
        fig.savefig(os.path.join(work_dir, f'predictions_metrics_{m}.png'), bbox_inches='tight')

    # compare metrics on calibration for al models
    calib_metrics = calib_results.columns[calib_results.columns.str.contains('CI')].str.split('_').str.get(0)
    for m in calib_metrics:
        fig = vis.plot_metrics(calib_results, metric=m, compare_by='feature', group_by='method', task=config.task)
        fig.savefig(os.path.join(work_dir, f'calibration_metrics_{m}.png'), bbox_inches='tight')

    # calibration diagrams (not for multi-task)
    if FLAGS.task_index != -1 or y_true.shape[-1] == 1:
        dia_results = pd.concat(dia_results, ignore_index=True)
        dia_results['method'] = dia_results['method'].astype('category').cat.set_categories(METHOD_ORDER)
        dia_results['feature'] = dia_results['feature'].astype('category').cat.set_categories(FEATURE_ORDER)
        dia_results = dia_results.sort_values(['method', 'feature'])

        # plot calibration curve for each method
        for method in np.unique(dia_results['method']):
            fig = vis.plot_calibration_diagram(dia_results[dia_results['method'] == method], compare_by='feature',
                                               label=method, task=config.task)
            fig.savefig(os.path.join(work_dir, f'calibration_diagram_{method}.png'), bbox_inches='tight')

    # plot scatter plot of performance vs calibration
    all_results = pd.concat([calib_results, pred_results], axis=1)
    if config.task == enums.TaskType.regression:
        perf_m, calib_m = 'R^2', 'absCVPParea'
    else:
        perf_m, calib_m = 'auroc', 'ece'
    fig = vis.plot_pred_calib_comparison(all_results, perf_m, calib_m)
    fig.savefig(os.path.join(work_dir, 'perfvcalib.png'), bbox_inches='tight')


if __name__ == '__main__':
    flags.mark_flag_as_required('dataset')
    app.run(main)
