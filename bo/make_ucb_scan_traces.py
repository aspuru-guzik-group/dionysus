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

def unpack_preds(res, model_type, feat_type, acq_type, beta):
    p_metrics = []
    c_metrics = []
    for seed in range(len(res)):
        r_pred = res[seed]['pred_data']
        r_pred_p = [d['p_metric'] for d in r_pred if d]
        r_pred_c = [d['c_metric'] for d in r_pred if d]

        # performance metrics
        r_df_p = pd.DataFrame(r_pred_p)
        r_df_p['batch'] = np.arange(r_df_p.shape[0])+1
        r_df_p['seed'] = seed
        r_df_p['model'] = model_type
        r_df_p['feature'] = feat_type
        r_df_p['acq_func'] = acq_type
        r_df_p['beta'] = beta

        # calibration metrics
        r_df_c = pd.DataFrame(r_pred_c)
        r_df_c['batch'] = np.arange(r_df_c.shape[0])+1
        r_df_c['seed'] = seed
        r_df_c['model'] = model_type
        r_df_c['feature'] = feat_type
        r_df_c['acq_func'] = acq_type
        r_df_c['beta'] = beta
        
        p_metrics.append(r_df_p)
        c_metrics.append(r_df_c)
        
    return (
        pd.concat(p_metrics, ignore_index=False),
        pd.concat(c_metrics, ignore_index=False)
    )

def get_bayes_opt_data(res):
    
    return df_traces

def get_scan_data(fname):
    res = pickle.load(open(fname, 'rb'))
    info = fname.split('/')[3].split('-')
    dataset = info[0]
    model = info[2]
    feature = info[3]
    acq_func = info[4]
    beta = info[-1]

    res_p_metrics, res_c_metrics = unpack_preds(res, model, feature, acq_func, beta)

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
        df['seed'] = [j]*len(trace)
        df_traces.append(df)
    
    df_traces = pd.concat(df_traces).reset_index().rename(columns={'traces': 'Target', 'index': 'Evaluations'})
    df_traces['feature'] = [feature]*len(df_traces)
    df_traces['beta'] = [beta]*len(df_traces)
    # df_traces = pd.concat(df_traces, ignore_index = False)

    return res_p_metrics, res_c_metrics, df_traces
    


def main(_):
    dataset = FLAGS.dataset
    DATA_DIR = os.path.join('cluster_submission', 'ucb_scan', dataset)

    FEATURE_ORDER = enums.VECTOR_FEATURES + enums.GRAPH_FEATURES
    FEATURE_ORDER.remove('graphembed')
    METHOD_ORDER = list(enums.Models)

    vis.plotting_settings()
    paths = glob.glob(os.path.join(DATA_DIR, '*', 'results.pkl'))
    df_all = utils.parallel_map(get_scan_data, paths)

    p_metrics, c_metrics, opt_traces = [], [], []
    for df in df_all:
        p_metrics.append(df[0])
        c_metrics.append(df[1])
        opt_traces.append(df[2])

    p_metrics = pd.concat(p_metrics, ignore_index=True)
    c_metrics = pd.concat(c_metrics, ignore_index=True)
    opt_traces = pd.concat(opt_traces, ignore_index=True)

    p_metrics['feature'] = p_metrics['feature'].astype('category').cat.set_categories(FEATURE_ORDER).cat.remove_unused_categories()
    c_metrics['feature'] = c_metrics['feature'].astype('category').cat.set_categories(FEATURE_ORDER).cat.remove_unused_categories()
    opt_traces['feature'] = opt_traces['feature'].astype('category').cat.set_categories(FEATURE_ORDER).cat.remove_unused_categories()

    p_metrics = p_metrics.sort_values('beta')
    c_metrics = c_metrics.sort_values('beta')
    opt_traces = opt_traces.sort_values('beta')

    # access dataset
    work_dir = os.path.join(gpmol.files.get_data_dir(ROOT_DIR), dataset)
    _, config, _ = gpmol.datasets.load_dataset(dataset, work_dir)
    config.task = config.tasks[0]
    smi_split, _, y = gpmol.datasets.load_task(dataset, enums.FeatureType.mfp, enums.Models.sngp, work_dir, task_index=-1)
            
    all_models = p_metrics['model'].unique().tolist()

    if config.task == gpmol.enums.TaskType.regression:
        p_m = r'$R^2$'
        c_m = 'Absolute miscalibration area'
        p_metrics = p_metrics.rename(columns={'R^2': p_m, 'batch': 'Batches'})
        c_metrics = c_metrics.rename(columns={'absCVPParea': c_m, 'batch': 'Batches'})
    elif config.task == gpmol.enums.TaskType.binary:
        p_m = 'AUROC'
        c_m = 'Expected calibration error'
        p_metrics = p_metrics.rename(columns={'auroc': p_m, 'batch': 'Batches'})
        c_metrics = c_metrics.rename(columns={'ece': c_m, 'batch': 'Batches'})
    else:
        raise NotImplementedError('No such task.')

    # plot performance trace
    target_name = 'Log Solubility\n(log[mol/L])' # 'Free energy (kcal/mol)' 

    NUM_INIT = int(0.05*len(smi_split.train))
    if NUM_INIT < 25:
        NUM_INIT = 25
    elif NUM_INIT > 100:
        NUM_INIT = 100

    best = min(y.train)
    opt_traces = opt_traces.rename(columns={'Target': target_name})
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax = ax.flatten()
    for m, a in zip(all_models, ax):
        plot_df = opt_traces[opt_traces['model']==m].copy()
        plot_df['feature'] = plot_df['feature'].cat.remove_unused_categories()
        sns.lineplot(data=plot_df, x='Evaluations', y=target_name, hue='beta', ax = a, palette='rocket_r')
        a.set_title(m)
        a.set_box_aspect(1)
        a.axvline(NUM_INIT, c='k', ls=':')
        if config.task == enums.TaskType.regression:
            a.axhline(best, c='k', ls='--', alpha=0.8)
    plt.savefig(os.path.join(DATA_DIR, f'opt_trace_ucb_scan.png'), bbox_inches='tight')

    # plot the performance metrics as they evolve
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax = ax.flatten()
    for m, a in zip(all_models, ax):
        plot_df = p_metrics[p_metrics['model']==m].copy()
        plot_df['feature'] = plot_df['feature'].cat.remove_unused_categories()
        # sns.lineplot(data=plot_df, x='Evaluations', y=target_name, hue='feature', ax=a)
        sns.lineplot(data=plot_df, x='Batches', y=p_m, hue='beta', ax = a, palette='rocket_r')
        a.set_title(m)
        a.set_box_aspect(1)
    plt.savefig(os.path.join(DATA_DIR, f'performance_ucb_scan.png'), bbox_inches='tight')

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax = ax.flatten()
    for m, a in zip(all_models, ax):
        plot_df = c_metrics[c_metrics['model']==m].copy()
        plot_df['feature'] = plot_df['feature'].cat.remove_unused_categories()
        g = sns.lineplot(data=plot_df, x='Batches', y=c_m, hue='beta', ax = a, palette='rocket_r')
        g.get_legend().remove()
        a.set_title(m)
        a.set_box_aspect(1)

    plt.savefig(os.path.join(DATA_DIR, f'calibration_ucb_scan.png'), bbox_inches='tight')



if __name__ == '__main__':
    flags.mark_flag_as_required('dataset')
    app.run(main)