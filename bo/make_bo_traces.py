import os, sys

ROOT_DIR = '..'
sys.path.append(ROOT_DIR)

import pandas as pd
import numpy as np
import pickle
import glob
import seaborn as sns
import matplotlib.pyplot as plt

import gpmol
from gpmol import vis, enums, utils
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'Dataset to process')
# flags.DEFINE_integer('task_index', -1, 'Select task for multi-target dataset. Set -1 to use all.')

def get_bayes_opt_data(fname):
    res = pickle.load(open(fname, 'rb'))
    info = fname.split('/')[2].split('-')
    dataset = info[0]
    model = info[2]
    feature = info[3]
    acq_func = info[4]
    # model = info[1]
    # feature = info[2]
    # acq_func = info[3]

    df_traces = []
    for j, df in enumerate(res):
        df = df['opt_data']
        trace = []
        for i, r in df.iterrows():
            i += 1
            if i == 1:
                goal = r['goal']
                acq_func = r['acq_func']

            # if regression ucb
            if acq_func == 'ucb':
                if i == 1:
                    mem = r['y']
                else:
                    compare_fn = np.maximum if goal == 'maximize' else np.minimum
                    mem = compare_fn(r['y'], mem)
                trace.append(mem)

            # if binary greedy
            elif acq_func == 'greedy_binary':
                if i == 1:
                    mem = r['y']
                else:
                    mem = trace[-1] + r['y']
                trace.append(mem)

        df['traces'] = trace
        df['run'] = [j]*len(trace)
        df_traces.append(df)
    
    df_traces = pd.concat(df_traces).reset_index().rename(columns={'traces': 'Target', 'index': 'Evaluations'})
    df_traces['feature'] = [feature]*len(df_traces)
    
    return df_traces
    


def main(_):
    dataset = FLAGS.dataset
    DATA_DIR = os.path.join('cluster_submission', dataset)

    FEATURE_ORDER = enums.VECTOR_FEATURES + enums.GRAPH_FEATURES
    FEATURE_ORDER.remove('graphembed')
    # METHOD_ORDER = list(enums.Models) + ['random']

    vis.plotting_settings()
    paths = glob.glob(os.path.join(DATA_DIR, '*', 'results.pkl'))
    df_all = utils.parallel_map(get_bayes_opt_data, paths)

    df_all = pd.concat(df_all, ignore_index=True)
    df_all['feature'] = df_all['feature'].astype('category').cat.set_categories(FEATURE_ORDER).cat.remove_unused_categories()
    # df_all['model'] =  df_all['model'].astype('category').cat.set_categories(METHOD_ORDER).cat.remove_unused_categories()

    # access dataset
    work_dir = os.path.join(gpmol.files.get_data_dir(ROOT_DIR), dataset)
    _, config, _ = gpmol.datasets.load_dataset(dataset, work_dir)
    config.task = config.tasks[0]
    smi_split, _, y = gpmol.datasets.load_task(dataset, enums.FeatureType.mfp, enums.Models.sngp, work_dir, task_index=-1)

    init_frac = 0.05 if config.task == enums.TaskType.regression else 0.10
    NUM_INIT = int(init_frac*len(smi_split.train))
    if NUM_INIT < 25:
        NUM_INIT = 25
    elif NUM_INIT > 100:
        NUM_INIT = 100

    goal = df_all['goal'].loc[0]
    if config.task == enums.TaskType.regression:
        if goal == 'maximize':
            best = np.max(y.train)
            threshold = np.percentile(y.values, 90)
            compare_fn = np.greater
        else:
            best = np.min(y.train)
            threshold = np.percentile(y.values, 10)
            compare_fn = np.less
        target_name = config.target_columns[0]
        target_name = 'Free energy\n(kcal/mol)' # 'Log Solubility\n(log[mol/L])'  #  # 'Half life\n(log[days])'

    elif config.task == enums.TaskType.binary:
        best = y.values.sum()/len(y.values)
        target_name = 'Fraction of positives'
        df_all['Target'] = df_all['Target'] / y.values.sum()
            
    # all_models = df_all['model'].unique().tolist()
    all_models = ['bnn', 'ngboost', 'gp', 'sngp/gnngp']
    # all_models.remove('random')
    # all_models.remove('gnngp')
    # all_models.remove('sngp')
    # all_models.append('sngp/gnngp')

    df_all = df_all.rename(columns={'Target': target_name})

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    ax = ax.flatten()
    for m, a in zip(all_models, ax):
        if '/' in m:
            m1, m2 = m.split('/')
            plot_df = df_all[(df_all['model']==m1) | (df_all['model']==m2)].copy()
        else:
            plot_df = df_all[df_all['model']==m].copy()
        plot_df['feature'] = plot_df['feature'].cat.remove_unused_categories()
        sns.lineplot(data=plot_df, x='Evaluations', y=target_name, hue='feature', ax=a, palette=vis.CMAP)
        g = sns.lineplot(data=df_all[df_all['model']=='random'], x='Evaluations', y=target_name, hue='model', ax=a, palette=['k'])
        # g.get_legend().set_title(No
        g.get_legend().remove()
        a.axvline(NUM_INIT, c='k', ls=':')
        if config.task == enums.TaskType.regression:
            a.axhline(best, c='k', ls='--', alpha=0.8)
        a.set_title(m)
        # a.set_box_aspect(1)
        # legend.texts[0].set_text('')
    plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
    plt.savefig(os.path.join(DATA_DIR, f'{dataset}_trace.png'), bbox_inches='tight')

    # calculate statistics
    df_stats = df_all.copy()
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
                per_run.append(gggrp_df['hit'].sum())
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

    pd.DataFrame(results).to_csv(os.path.join(DATA_DIR, f'{dataset}_bo_results.csv'), index=False)
    # for i, m in enumerate(np.unique(df_all['model']):
    #     sns.lineplot(data = df_all, x = 'Evaluations', y = 'Target',  ax=ax[i])



if __name__ == '__main__':
    flags.mark_flag_as_required('dataset')
    app.run(main)