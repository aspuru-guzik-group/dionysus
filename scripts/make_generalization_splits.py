"""Train and make all predictions."""
import os
import sys
import warnings

warnings.simplefilter("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT_DIR = '..'
sys.path.append(ROOT_DIR)

import numpy as np
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import dionysus
from dionysus import splits
from skmultilearn.model_selection import IterativeStratification

dionysus.vis.plotting_settings()

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'Dataset to process')
flags.DEFINE_integer('seed', 0, 'Random seed. Default 0.')
flags.DEFINE_integer('task_index', -1, 'Select task for multi-target dataset. All tasks set to -1.')
flags.DEFINE_integer('splits_per_level', 3, 'How many splits to subsample per combination level')
flags.DEFINE_float('val_size', .15, 'Amount of data to use for validation.', lower_bound=0.0, upper_bound=1.0)


def main(_):
    SEED, DATASET, TASK_INDEX, SAMPLE_SPLITS, VAL_SIZE = FLAGS.seed, FLAGS.dataset, FLAGS.task_index, FLAGS.splits_per_level, FLAGS.val_size
    # set random seed
    dionysus.utils.set_random_seed(SEED)

    work_dir = os.path.join(dionysus.files.get_data_dir(ROOT_DIR), DATASET)
    results_dir = os.path.join(work_dir, 'generalization')
    _, config, _ = dionysus.datasets.load_dataset(DATASET, work_dir)
    config.task = config.tasks[TASK_INDEX]
    if not os.path.exists(results_dir):
        print(f"Creating directory {results_dir}")
        os.makedirs(results_dir)

    at_work_dir = lambda fname: os.path.join(work_dir, fname)
    at_results_dir = lambda fname: os.path.join(results_dir, fname)

    FEAT = dionysus.enums.FeatureType.mfp
    smi_split, x_split, y_split = dionysus.datasets.load_task(DATASET, FEAT, dionysus.enums.Models.ngboost, work_dir,
                                                           task_index=TASK_INDEX, mask_inputs=False)
    smi, x, y = smi_split.values, x_split.values, y_split.values
    mols = np.array([dionysus.chem.smi_to_mol(s) for s in smi_split.values])

    # train clusterer and get the labels
    x_reduced, cluster_labels = splits.get_cluster_labels(x, random_state=SEED)
    np.savez_compressed(at_results_dir('cluster_info.npz'),
                        cluster_labels=cluster_labels, x_reduced=x_reduced, smi=smi)

    # # get the clusters
    cluster_splits = splits.get_tvt_cluster_splits(smi, y, cluster_labels, config.task, n_samples=SAMPLE_SPLITS)
    for s in cluster_splits:
        del s['name']
    smi_test = s['test']      # store the smiles used for test set

    # save each cluster
    for i, split_dict in enumerate(cluster_splits):
        fname = dionysus.files.get_filename(results_dir, 'split', [f'cluster{i}', 'tvt'])
        np.savez_compressed(fname, **split_dict)

    print('Start generating cluster plots')
    dionysus.vis.plotting_settings()
    cluster_cmap = dionysus.vis.get_cluster_cmap(np.max(cluster_labels))
    dionysus.vis.plot_cluster_index(cluster_labels, DATASET, cmap=cluster_cmap)
    dionysus.vis.save_figure(work_dir, 'cluster_index')
    plt.clf()
    dionysus.vis.plot_cluster_space(x_reduced, cluster_labels, DATASET, cmap=cluster_cmap)
    dionysus.vis.save_figure(work_dir, 'cluster_umap')
    plt.clf()
    dionysus.vis.plot_cluster_counts(cluster_labels, DATASET, cmap=cluster_cmap)
    dionysus.vis.save_figure(work_dir, 'cluster_counts')
    plt.clf()
    svg = dionysus.vis.draw_cluster_mols(mols, cluster_labels, 3)
    dionysus.vis.save_svg(svg, work_dir, 'cluster_mols')

    print('Make diverse tvt split')
    fps = dionysus.chem.manysmi_to_fps(smi)
    joint_labels = splits.build_joint_labels(fps, y, config.task)
    trainval = np.logical_not(np.isin(smi, smi_test))
    smi_trainval = smi[trainval]
    labels_trainval = joint_labels[trainval]
    train, val = splits.get_it_split(labels_trainval, train_size=1.0 - VAL_SIZE)
    fname = dionysus.files.get_filename(work_dir, 'split', ['diverse', 'tvt'])
    np.savez_compressed(fname, train=smi_trainval[train], val=smi_trainval[val], test=smi_test)

    print('Make incremental diverse tvt splits')
    splitter = IterativeStratification(n_splits=8, order=2)
    pieces = [val for _, val in splitter.split(smi_trainval, labels_trainval)]
    picks = pieces.pop()
    for index, piece in enumerate(pieces):
        picks = np.concatenate((picks, piece))
        smi_picks = smi_trainval[picks]
        label_picks = joint_labels[picks]
        train, val = splits.get_it_split(label_picks, train_size=1.0 - VAL_SIZE)
        name = f'incremental-diverse{index}'
        fname = dionysus.files.get_filename(results_dir, 'split', [name, 'tvt'])
        np.savez_compressed(fname, **{'train': smi_picks[train],
                                      'val': smi_picks[val],
                                      'test': smi_split.test
                                      })

    print('Make incremental standard tvt splits')
    splitter = splits.get_default_kfold_splitter(n_splits=8, task_type=config.task)
    y_trainval = y[trainval]
    pieces = [val for _, val in splitter.split(smi_trainval, y_trainval)]
    picks = pieces.pop()
    for index, piece in enumerate(pieces):
        picks = np.concatenate((picks, piece))
        split_fn = splits.get_default_split_fn(y_trainval[picks], config.task)
        train, val = split_fn(picks, train_size=1.0 - VAL_SIZE)
        name = f'incremental-standard{index}'
        fname = dionysus.files.get_filename(results_dir, 'split', [name, 'tvt'])
        np.savez_compressed(fname, **{'train': smi_trainval[train],
                                      'val': smi_trainval[val],
                                      'test': smi_split.test
                                      })


if __name__ == '__main__':
    flags.mark_flag_as_required('dataset')
    app.run(main)
