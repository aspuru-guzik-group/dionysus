import glob
from typing import Any, Callable, List

import dask
import dask.bag as db
import dask.diagnostics
import numpy as np
import pandas as pd
import tensorflow as tf

from . import files

RANDOM_SEED = 42


def get_matching_files(fname: str) -> List[str]:
    return [name for name in glob.glob(fname)]


def set_random_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f'Random seed set to {seed}')


def print_module_versions(module_list):
    """Print module versions"""
    for module in module_list:
        print(f'{module.__name__:<20s}: {module.__version__}')


def parallel_map(func: Callable[[Any], Any], data: List[Any], use_pbar: bool = True) -> List[Any]:
    """Parallelize a function applied to a list using dask."""
    seq_bag = db.from_sequence(data)
    if use_pbar:
        pbar = dask.diagnostics.ProgressBar()
        pbar.register()
    values = seq_bag.map(func).compute()
    if use_pbar:
        pbar.unregister()
    return values


class SmilesMap:

    def __init__(self, fname: str, values: str = 'values', key: str = 'isosmiles'):
        data = files.load_npz(fname)
        assert key in data, f'{key} not found in {data.keys()}'
        assert values in data, f'{values} not found in {data.keys()}'
        self.mapper = pd.Series(data=np.arange(len(data[key])), index=data[key])
        self.values = data[values]
        # Check for duplicates.
        dups = self.mapper.index[self.mapper.index.duplicated()].tolist()
        if dups:
            raise ValueError(f'Found {len(dups)} duplicated isosmiles={dups}')

    def __call__(self, key_array: np.ndarray) -> np.ndarray:
        assert isinstance(key_array, np.ndarray), 'expected np.ndarray as input!'
        return self.values[self.mapper[key_array].values]
