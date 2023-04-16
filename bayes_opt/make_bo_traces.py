import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT_DIR = '..'
sys.path.append(ROOT_DIR)

import pandas as pd
import numpy as np
import pickle
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import dionysus
from dionysus import vis, enums, utils
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'Dataset to process')

FEATURE_ORDER = enums.VECTOR_FEATURES + enums.GRAPH_FEATURES
FEATURE_ORDER.remove('graphembed')
METHOD_ORDER = list(enums.Models) + ['random', '1-nn']

tname_map = {
    'delaney': 'Log Solubility\n(log[mol/L])',
    'opera_BioHL': 'Half life\n(log[days])',
    'freesolv': 'Free energy\n(kcal/mol)'
}


def main(_):
    vis.plotting_settings()
    work_dir = os.path.join(dionysus.files.get_data_dir(ROOT_DIR), FLAGS.dataset)
    data_dir = os.path.join(work_dir, 'bayesian_optimization')

    _, config, _ = dionysus.datasets.load_dataset(FLAGS.dataset, work_dir)
    config.task = config.tasks[0]
    smi_split, _, y = dionysus.datasets.load_task(FLAGS.dataset, enums.FeatureType.mfp, 
        enums.Models.sngp, work_dir, task_index=-1)
    
    trace_files = dionysus.utils.get_matching_files(
        dionysus.files.get_filename(data_dir, 'bo_traces', ['*', '*'])
    )

    # go through files and create results for plotting
    trace_results = []
    for fname in trace_files:
        df = pd.read_csv(fname)
        trace_results.append(df)

    # append and set categorical order
    trace_results = pd.concat(trace_results)
    trace_results['feature'] = trace_results['feature'].astype('category').cat.set_categories(FEATURE_ORDER).cat.remove_unused_categories()
    trace_results[trace_results['model'] == 'nn'] = trace_results[trace_results['model'] == 'nn'].assign(model='1-nn')      # rename nearest-neighbour

    # assumes default design choice
    init_frac = 0.05 if config.task == enums.TaskType.regression else 0.10
    NUM_INIT = int(init_frac*len(smi_split.train))
    if NUM_INIT < 25:
        NUM_INIT = 25
    elif NUM_INIT > 100:
        NUM_INIT = 100
    
    goal = trace_results['goal'].iloc[0]            # assumes you do not have conflicting goals
    working_set = np.append(y.train, y.val, axis=0)
    if config.task == enums.TaskType.regression:
        if goal == 'maximize':
            best = np.max(working_set)
            threshold = np.percentile(working_set, 90)
            compare_fn = np.greater_equal
            norm = np.sum(working_set >= threshold)
        else:
            best = np.min(working_set)
            threshold = np.percentile(working_set, 10)
            compare_fn = np.less_equal
            norm = np.sum(working_set <= threshold)
        target_name = tname_map[config.name]        # just a nicer name for the labels

    elif config.task == enums.TaskType.binary:
        best = working_set.sum()/len(working_set)
        target_name = 'Fraction of positives'
        trace_results['trace'] = trace_results['trace'] / working_set.sum()
        norm = working_set.sum()
    trace_results = trace_results.rename(columns={
        'trace': target_name,
        'eval': 'Evaluations',
    })
            
    all_models = ['bnn', 'ngboost', 'gp', 'sngp/gnngp']     # append sngp/gnngp

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    ax = ax.flatten()
    for m, a in zip(all_models, ax):
        if '/' in m:
            m1, m2 = m.split('/')
            plot_df = trace_results[(trace_results['model']==m1) | (trace_results['model']==m2)].copy()
        else:
            plot_df = trace_results[trace_results['model']==m].copy()
        plot_df['feature'] = plot_df['feature'].cat.remove_unused_categories()
        sns.lineplot(data=plot_df, x='Evaluations', y=target_name, hue='feature', ax=a, palette=vis.CMAP)
        g = sns.lineplot(data=trace_results[trace_results['model']=='random'], x='Evaluations', y=target_name, hue='model', ax=a, palette=['k'])
        g = sns.lineplot(data=trace_results[trace_results['model']=='1-nn'], x='Evaluations', y=target_name, hue='model', ax=a, palette=['tab:gray'])
        
        g.get_legend().remove()
        if config.task == enums.TaskType.regression:
            a.axhline(best, c='k', ls='--', alpha=0.8)
        a.set_title(m)
        a.set_xlim(0, None)

    plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))

    for a in ax:
        y_min, y_max = a.get_ylim()
        a.add_patch(
            Rectangle((0, y_min), NUM_INIT, y_max-y_min,
            facecolor='y',
            fill=True,
            alpha=0.1)
        )

    plt.savefig(os.path.join(data_dir, f'{FLAGS.dataset}_trace.png'), bbox_inches='tight')

    # calculate statistics
    df_stats = trace_results.copy()
    df_stats['Evaluations'] = df_stats['Evaluations'] - NUM_INIT
    df_stats = df_stats[df_stats['Evaluations'] >= 0]

    results = {'mean_hit': [], 'std_hit': [], 'ci_hit': [], 'method': [], 'feature': []}
    
    for meth, grp_df in df_stats.groupby('model'):
        for feat, ggrp_df in grp_df.groupby('feature'):
            per_run = []
            for _, gggrp_df in ggrp_df.groupby('run'):
                if config.task == enums.TaskType.regression:
                    gggrp_df['hit'] = gggrp_df['y'].apply(lambda r: compare_fn(r, threshold).astype(int))
                elif config.task == enums.TaskType.binary:
                    gggrp_df['hit'] = gggrp_df['y']
                per_run.append(gggrp_df['hit'].sum() / norm)
            if len(per_run) == 0:
                results['mean_hit'].append(0)
                results['std_hit'].append(0)
                results['ci_hit'].append(0)
            else:
                results['mean_hit'].append(np.mean(per_run))
                results['std_hit'].append(np.std(per_run))
                results['ci_hit'].append(np.std(per_run)/np.sqrt(len(per_run))*1.96)
            results['method'].append(meth)
            results['feature'].append(feat)
            print(f'{meth} + {feat}: {len(per_run)}')

    pd.DataFrame(results).to_csv(os.path.join(data_dir, f'{FLAGS.dataset}_bo_results.csv'), index=False)
    # for i, m in enumerate(np.unique(df_all['model']):
    #     sns.lineplot(data = df_all, x = 'Evaluations', y = 'Target',  ax=ax[i])



if __name__ == '__main__':
    flags.mark_flag_as_required('dataset')
    app.run(main)
