"""Preprocess datasets from a db registry and generated processed, ML-ready."""
import os
import sys
from typing import Dict, Any, List

ROOT_DIR = '..'
sys.path.append(ROOT_DIR)
import ml_collections
import dataclasses
import numpy as np
from absl import app, flags
import pandas as pd
import csv

import dionysus

DATA_DIR = dionysus.files.get_data_dir(ROOT_DIR)
PROCESSED_DIR = dionysus.files.get_processed_dir(ROOT_DIR)
RAW_DIR = os.path.join(DATA_DIR, 'raw')
DB_FILE = os.path.join(DATA_DIR, 'dataset_registry_raw.csv')

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', None, 'Dataset to process')
flags.DEFINE_integer('rare_atom_threshold', 0,
                     'Remove molecules with rare atoms')
flags.DEFINE_bool('test_mode', False, 'Testing mode for preprocessing.')


def get_db_info(dataset: str) -> Dict[str, Any]:
    db_df = pd.read_csv(os.path.join(DATA_DIR, DB_FILE), quotechar='"')
    datasets = list(db_df['shortname'].unique())
    match_df = db_df[db_df['shortname'] == dataset]
    assert len(
        match_df
    ) == 1, F'Expected only one match for {dataset}, found {len(match_df)}, possible values are {datasets}'
    return match_df.iloc[0].to_dict()


def format_columns_txt(txt: str, df: pd.DataFrame) -> List[str]:
    cols = [l for l in csv.reader([txt])][0]
    all_cols = df.columns.tolist()
    not_found = [c for c in cols if c not in all_cols]
    if not_found:
        raise ValueError(
            f'Did not find {not_found} in dataframe columns {all_cols}')
    return cols


def main(_):
    name = FLAGS.dataset
    info = get_db_info(name)
    print(f'Processing {name}')
    print(f'info: {info}')
    info['task_types'] = [dionysus.enums.TaskType(i.strip()) for i in info['task_types'].split(',')]
    fname = os.path.join(RAW_DIR, info['filename'])
    df = pd.read_csv(fname, quotechar='"')
    df.columns = df.columns.str.strip()
    mol_col, smi_col = 'mol', info['smiles_column']
    assert mol_col not in df.columns.tolist(
    ), f" 'mol' is in datafile {fname}, please rename "
    assert smi_col in df.columns.tolist(
    ), f"Did not find '{smi_col}' column' in {fname}"

    df[mol_col] = df[smi_col].apply(dionysus.chem.smi_to_mol)
    not_valid = df[mol_col].isnull()
    print(f'Found {not_valid.sum()} not valid molecules')
    df = df[np.logical_not(not_valid)]
    not_multi = df[mol_col].apply(dionysus.chem.is_single)
    print(f'Found {np.logical_not(not_multi).sum()} multi-component molecules')
    df = df[not_multi]
    many_atoms = df[mol_col].apply(dionysus.chem.is_larger_molecule)
    print(f'Found {np.logical_not(many_atoms).sum()} single-atom molecules')
    df = df[many_atoms]
    df['isosmiles'] = df[mol_col].apply(dionysus.chem.get_isosmiles)
    natoms = df[mol_col].apply(lambda m: m.GetNumAtoms()).values
    natoms_range = (int(np.min(natoms)), int(np.max(natoms)))
    n_mols = len(df['isosmiles'].unique())

    has_unique_smiles = n_mols == len(df)

    atom_set = dionysus.chem.get_atom_set(df[mol_col])
    print(f'Found the atom_set = {atom_set}')

    relevant_columns = []
    if info['id_column'] != 'N/A (rows)':

        relevant_columns.extend(format_columns_txt(info['id_column'], df))
    else:
        df = df.reset_index()
        relevant_columns.append('index')
    relevant_columns.append(['isosmiles'])
    target_columns = format_columns_txt(info['target_column(s)'], df)
    relevant_columns.append(target_columns)
    fname = os.path.join(PROCESSED_DIR, f'{name}.csv')
    df.to_csv(fname, index=False)

    new_info = dionysus.datasets.DatasetInfo(name=name,
                                          atom_set=atom_set,
                                          unique_smiles=has_unique_smiles,
                                          natom_range=natoms_range,
                                          has_salts=False,
                                          n_datapoints=len(df),
                                          n_mols=n_mols,
                                          target_columns=target_columns,
                                          tasks=info['task_types'],
                                          doi=info['reference '])
    new_info = ml_collections.ConfigDict(dataclasses.asdict(new_info))
    fname = os.path.join(PROCESSED_DIR, f'{name}.json')
    with open(fname, 'w') as afile:
        afile.write(new_info.to_json())

    # write to new directory
    new_data_dir = os.path.join(DATA_DIR, name)
    if not os.path.exists(new_data_dir):
        os.makedirs(new_data_dir)
    fname = os.path.join(new_data_dir, f'{name}.csv')
    df.to_csv(fname, index=False)
    fname = os.path.join(new_data_dir, f'{name}.json')
    with open(fname, 'w') as afile:
        afile.write(new_info.to_json())


if __name__ == '__main__':
    flags.mark_flag_as_required('dataset')
    app.run(main)
