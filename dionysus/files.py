import json
import os
from typing import Dict, Optional, Union, Sequence, Tuple

import graph_nets
import ml_collections
import numpy as np

from . import types, graphs, enums


def get_data_dir(root_dir: str) -> str:
    return os.path.join(root_dir, 'data')


def get_processed_dir(root_dir: str) -> str:
    return os.path.join(get_data_dir(root_dir), 'processed')


def get_results_dir(root_dir: str) -> str:
    return os.path.join(root_dir, 'results')


FILES_MAP = {'mol': '{}_mols.sdf.lzma',
             'predictions': 'predictions_{}_{}.npz',
             #  'uncertainties': '{}_{}_uncertainties.npz',
             #  'embeddings': '{}_embeddings.npz',
             'graphs': 'graph_tuples_{}.npz',
             'feature': 'features_{}.npz',
             'split': '{}_{}_smiles.npz',
             'config': '{}.json',
             'data': '{}.csv',
             #  'embeds': '{}_{}_embedded.npz',
             'model_weights': '{}_{}_model/weights',
             'model': '{}_{}_model',
             'viz': '{}',
             'bo_traces': 'bo_traces_{}_{}.csv',
             'bo_metrics': 'bo_metrics_{}_{}.csv'
             }


def get_filename(work_dir: str, ftype: str, fname_args: Optional[Union[Sequence[str], str]] = None) -> str:
    fname = FILES_MAP.get(ftype)
    if fname is None:
        raise ValueError(f'{ftype} not found in FILES_MAP ({list(FILES_MAP.keys())})')
    fname_args = fname_args or []
    fname_args = [fname_args] if isinstance(fname_args, str) else fname_args
    n_format = fname.count('{')
    if n_format != len(fname_args):
        raise ValueError(
            f'{ftype} is formatable ({fname}), expecting {n_format} arguments, found {len(fname_args)} ({fname_args}).'
        )
    if n_format != fname.count('}'):
        raise ValueError(
            f'{ftype} has inconsistent number of parenthesis ({fname})')
    return os.path.join(work_dir, fname.format(*fname_args))


def get_file_info(fname: str, delimiter: str = '_'):
    ''' Load file information from file name.
    '''
    fname = os.path.split(fname)[-1]
    fname = fname.split('.')[0]
    infos = fname.split(delimiter)

    # get the information
    feat = infos[-1]
    feat = enums.FeatureType(feat)
    method = infos[1]

    info = {'method': method, 'feature': str(feat)}
    return info


def _check_ext(fname: str, ext: str):
    if not fname.endswith(ext):
        raise ValueError(f'Expected extension "{ext}" in {fname}')


def load_npz(fname: str) -> Dict[str, np.ndarray]:
    _check_ext(fname, 'npz')
    data = np.load(fname)
    return {key: data[key] for key in data.keys()}


def load_config(fname: str) -> types.ConfigDict:
    _check_ext(fname, '.json')
    with open(fname, 'r') as afile:
        config = ml_collections.ConfigDict(json.load(afile))
    return config


def save_graphstuple(fname: str, x: types.GraphsTuple, isosmiles: types.TextArray):
    """Save a list of graphstuples with np.savez_compressed."""
    assert len(isosmiles) == len(x.globals), 'Expecting same number of graphs and smiles'
    np_x = graphs.cast_to_np(x)
    data_dict = {
        k: getattr(np_x, k)
        for k in graph_nets.graphs.ALL_FIELDS
        if getattr(np_x, k) is not None
    }
    data_dict['isosmiles'] = isosmiles
    np.savez_compressed(fname, **data_dict)


def load_graphstuple(fname: str) -> Tuple[types.TextArray, types.GraphsTuple]:
    """Load a list of graphstuples with np.load."""
    data = load_npz(fname)
    data_dict = {k: None if k not in data else data[k] for k in graph_nets.graphs.ALL_FIELDS}
    x = graphs.cast_to_tf(graph_nets.graphs.GraphsTuple(**data_dict))
    return data['isosmiles'], x
