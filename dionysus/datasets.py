import dataclasses
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

from . import files, enums, types, utils, chem, graphs


@dataclasses.dataclass
class DatasetInfo:
    name: str
    atom_set: str
    unique_smiles: bool
    natom_range: Tuple[int, int]
    has_salts: bool
    n_datapoints: int
    n_mols: int
    target_columns: List[str]
    tasks: List[enums.TaskType]
    doi: str
    smiles_column: str = 'isosmiles'
    task: Optional[enums.TaskType] = None

    def __post_init__(self):
        self.tasks = [enums.TaskType(i) for i in self.tasks]


def process_y(series: pd.Series, task: enums.TaskType) -> np.ndarray:
    if task == enums.TaskType.regression:
        if len(series.shape) == 1:
            return series.values.reshape(-1, 1)
        else:
            return series.values
    elif task == enums.TaskType.binary:
        series_np = series.to_numpy()
        # assert np.issubdtype(series_np.dtype, np.integer)
        # return label_binarize(series_np, classes=[0, 1])
        if len(series.shape) == 1:
            return series_np.reshape(-1, 1).astype(int)
        else:
            return series_np.astype(int)
    else:
        raise ValueError(f'{task} not implemented!')


def load_dataset(name: enums.Dataset, work_dir: str) -> Tuple[pd.DataFrame, types.ConfigDict, types.TextArray]:
    name = enums.Dataset(name)
    config = files.load_config(files.get_filename(work_dir, 'config', name))
    config = DatasetInfo(**config)
    df = pd.read_csv(files.get_filename(work_dir, 'data', name))
    df = df.drop_duplicates(config.smiles_column).reset_index(drop=True)
    df['mol'] = df[config.smiles_column].apply(chem.smi_to_mol)
    smiles = np.array(df[config.smiles_column].tolist())
    return df, config, smiles


def load_task(name: enums.Dataset, feature_set: enums.FeatureType, model: enums.Models,
              work_dir: str, task_index: Optional[int] = 0, result_dir: str = None,
              split_type: Optional[str] = None, seed: int = 0,
              mask_inputs: bool = True, verbose: bool = True) -> Tuple[types.ArraySplit, types.DataSplit, types.ArraySplit]:
    feature_set = enums.FeatureType(feature_set)
    df, config, smiles = load_dataset(name, work_dir)
    config.task = config.tasks[task_index]

    # check if separate directory used for results and splits
    if result_dir is None:
        result_dir = work_dir

    # dealing with multilabel tasks
    if task_index == -1:
        y = process_y(df[config.target_columns[:]], config.task)
    else:
        y = process_y(df[config.target_columns[task_index]], config.task)

    # deciding split type
    if split_type is None:
        split_type = 'random' if config.task == enums.TaskType.regression else 'stratified'

    smi_dict = files.load_npz(files.get_filename(result_dir, 'split', [split_type, 'tvt']))
    indices = {}
    for key, smi in smi_dict.items():
        indices[key] = np.array(df[np.isin(smiles, smi)].index.tolist())

    split = types.IndexSplit(**indices)
    smi_split = types.ArraySplit(smiles, split)

    # decide the scaling based on the feature, model and task
    feature_scaler, target_scaler = scaling_options(feature_set, model, config.task)

    if feature_set in enums.GRAPH_FEATURES:
        loaded_smi, g = files.load_graphstuple(files.get_filename(work_dir, 'graphs', enums.FeatureType.graphnet))
        smi_to_index = {s: index for index, s in enumerate(loaded_smi)}
        new_indices = np.array([smi_to_index[s] for s in smiles])
        g = graphs.get_graphs(g, new_indices)
        x = graphs.GraphSplit(g, split)
    else:
        feature = utils.SmilesMap(files.get_filename(work_dir, 'feature', feature_set))
        values = feature(smiles)
        if mask_inputs:
            mask = np.logical_and(np.sum(np.isnan(values), axis=0) == 0, np.std(values, axis=0) > 0.0)
            values = values[:, mask]
            if verbose:
                print(f'Masking {np.sum(np.logical_not(mask))} feature dims for {feature_set}')

        x = types.ScaledArraySplit(values, split, feature_scaler)

    y = types.ScaledArraySplit(y, split, target_scaler)

    return smi_split, x, y


def scaling_options(feature_set: enums.FeatureType, model: enums.Models, task: enums.TaskType):
    # decide whether to scale features/targets and using what type of scalers
    if model == enums.Models.ngboost:
        return None, None

    # target scaling necessary?
    if task == enums.TaskType.regression:
        targets_scaler = 'standard'
    else:
        targets_scaler = None

    # features scaling necessary?
    if feature_set == enums.FeatureType.mordred:
        features_scaler = 'quantile'
    # elif feature_set == enums.FeatureType.graphembed:
    #     features_scaler = 'standard'
    else:
        features_scaler = None

    return features_scaler, targets_scaler


def get_sample_weights(y_train, y_val, y_test):
    ''' Returns the sample weights for all sets. 
    Sets should be stratified, so class weights can be determined using
    only the y_train set.
    '''
    n = len(y_train)
    class_weight = {
        0: 1.0 - (y_train == 0).sum().astype(float) / n,
        1: 1.0 - (y_train == 1).sum().astype(float) / n
    }

    w_train = [class_weight[t] for t in np.squeeze(y_train)]
    w_val = [class_weight[t] for t in np.squeeze(y_val)]
    w_test = [class_weight[t] for t in np.squeeze(y_test)]

    return w_train, w_val, w_test
