"""Preprocess datasets from a db registry and generated processed, ML-ready."""
import logging
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import warnings

warnings.filterwarnings("ignore")

ROOT_DIR = '..'
sys.path.append(ROOT_DIR)

from typing import List, Callable, Any, Union
import itertools
from absl import app, flags

import mordred
import mordred.descriptors
import rdkit.Chem.AllChem as Chem

import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split

import dionysus
from dionysus import types

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', None, 'Dataset to process')
flags.DEFINE_bool('overwrite', False, 'Re-calculate and overwrite if files already exist')
flags.DEFINE_bool('verbose', True, 'Print out training progress.')
flags.DEFINE_integer('seed', 0, 'Random seed. Default 0.')
flags.DEFINE_integer('task_index', -1, 'Select task for multi-target dataset. All tasks set to -1.')


def check_exists(fname: str, overwrite: bool):
    if os.path.exists(fname) and not overwrite:
        print(f'{fname} already exists. Skipping...')
        return True
    else:
        return False


def write_feature_file(fname: str, smi: types.TextArray, inp: Union[List[types.Mol], List[Any]],
                       compute_fn: Callable[[types.Mol], Any], overwrite: bool):
    if check_exists(fname, overwrite):
        return

    if compute_fn is None:
        values = np.array([np.array(v) for v in inp]).astype(np.float32)
    else:
        values = dionysus.utils.parallel_map(compute_fn, inp, use_pbar=FLAGS.verbose)
        values = np.array([np.array(v) for v in values]).astype(np.float32)
    np.savez_compressed(fname, isosmiles=smi, values=values)
    print(f'Wrote {fname}')


def write_split_file(fname: str, smi: types.TextArray, test_size: float,
                     val_size: float, train_size: float, overwrite: bool,
                     seed: int = 0, stratify: np.array = None):
    assert sum([test_size, val_size, train_size]) == 1.0, 'Check your splits'
    if check_exists(fname, overwrite):
        return

    # split train, validation and test
    if stratify is not None:
        # multilabel data will require iterative stratification
        if stratify.shape[-1] > 1:
            smi_train, stratify_train, smi_test, _ = iterative_train_test_split(smi, stratify, test_size=test_size)
            smi_train, _, smi_val, _ = iterative_train_test_split(smi_train, stratify_train,
                                                                  test_size=val_size / (train_size + val_size))
        else:
            smi_train, smi_test, stratify_train, _ = train_test_split(smi, stratify, random_state=seed,
                                                                      test_size=test_size, stratify=stratify)
            smi_train, smi_val = train_test_split(smi_train, random_state=seed,
                                                  test_size=val_size / (train_size + val_size), stratify=stratify_train)
    else:
        smi_train, smi_test = train_test_split(smi, random_state=seed, test_size=test_size)
        smi_train, smi_val = train_test_split(smi_train, random_state=seed,
                                              test_size=val_size / (train_size + val_size))

    # split train and validation
    for a, b in itertools.combinations([smi_train, smi_val, smi_test], 2):
        assert np.intersect1d(a, b).size == 0, 'Found bad split!'  # ensure no intersection between splits
    np.savez_compressed(fname, train=smi_train, val=smi_val, test=smi_test)
    print(f'Wrote {fname}')


def main(_):
    # set random seed
    dionysus.utils.set_random_seed(FLAGS.seed)

    work_dir = os.path.join(dionysus.files.get_data_dir(ROOT_DIR), FLAGS.dataset)
    df, config, smiles = dionysus.datasets.load_dataset(FLAGS.dataset, work_dir)
    config.task = config.tasks[FLAGS.task_index]  # if multidimension, expected to be the same for dims
    uniq_mol = df['mol'].tolist()

    def get_filename(ftype: str, fname_args=None) -> str:
        return dionysus.files.get_filename(work_dir, ftype, fname_args)

    # write split files for dataset
    if config.task == dionysus.enums.TaskType.regression:
        split_type = 'random'
        fname = get_filename('split', [split_type, 'tvt'])
        if not check_exists(fname, FLAGS.overwrite):
            write_split_file(fname, smiles, test_size=0.2, val_size=0.1, train_size=0.7, overwrite=FLAGS.overwrite)
    elif config.task == dionysus.enums.TaskType.binary:
        split_type = 'stratified'
        fname = get_filename('split', [split_type, 'tvt'])
        if FLAGS.task_index == -1:
            targets = df[config.target_columns[:]]
        else:
            targets = df[config.target_columns[FLAGS.task_index]].tolist()
        if not check_exists(fname, FLAGS.overwrite):
            write_split_file(fname, smiles, test_size=0.2, val_size=0.1, train_size=0.7, overwrite=FLAGS.overwrite,
                             stratify=targets)
    else:
        raise NotImplementedError('No such task.')

    # get Morgan fingerprints
    fname = get_filename('feature', dionysus.enums.FeatureType.mfp)
    write_feature_file(fname, smiles, uniq_mol, lambda m: Chem.GetMorganFingerprintAsBitVect(m, 3), FLAGS.overwrite)

    # get mordred descriptors
    fname = get_filename('feature', dionysus.enums.FeatureType.mordred)
    calc = mordred.Calculator(mordred.descriptors, ignore_3D=True)
    write_feature_file(fname, smiles, uniq_mol, lambda m: calc(m)._values, FLAGS.overwrite)

    # get graph tuples
    smi_to_graph = dionysus.graphs.MolTensorizer(config.atom_set)
    fname = get_filename('graphs', dionysus.enums.FeatureType.graphnet)
    x = smi_to_graph(smiles)
    dionysus.files.save_graphstuple(fname, x, smiles)

    # get graph embeddings
    fname = get_filename('feature', dionysus.enums.FeatureType.graphembed)
    if not check_exists(fname, FLAGS.overwrite):
        embeddings = dionysus.features.get_graphnet_embeddings(work_dir, FLAGS.dataset, task_index=FLAGS.task_index, verbose=FLAGS.verbose)
        write_feature_file(fname, smiles, embeddings, None, FLAGS.overwrite)


if __name__ == '__main__':
    flags.mark_flag_as_required('dataset')
    app.run(main)
