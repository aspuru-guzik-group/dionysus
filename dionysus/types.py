import abc
import dataclasses
from typing import NamedTuple, Union, List, Dict, Callable, Any

import graph_nets
import ml_collections
import numpy as np
import sonnet as snt
import tensorflow as tf
from sklearn.preprocessing import QuantileTransformer, StandardScaler, FunctionTransformer

# Typing aliases.
ConfigDict = ml_collections.ConfigDict
TextArray = Union[List[str], np.ndarray]
Indices = Union[List[int], np.ndarray]
EvalDict = Dict[str, Union[str, float]]
Mol = 'rdkit.Chem.Mol'
Atom = 'rdkit.Chem.Atom'
Bond = 'rdkit.Chem.Bond'
GraphsTuple = graph_nets.graphs.GraphsTuple
LossFunction = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
Activation = Callable[[tf.Tensor], tf.Tensor]
CastableActivation = Union[str, Activation]
Module = snt.Module
ModuleMaker = Callable[[], Module]
LossFunction = Callable[..., tf.Tensor]
MetricFunction = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


class IndexSplit(NamedTuple):
    train: Indices
    val: Indices
    test: Indices


class DataSplit(abc.ABC):
    """Basic class to hold train/val/test split of data."""
    values: Any
    split: IndexSplit

    @property
    def train(self) -> Any:
        return self.get_data_split(self.split.train)

    @property
    def val(self) -> Any:
        return self.get_data_split(self.split.val)

    @property
    def test(self) -> Any:
        return self.get_data_split(self.split.test)

    @abc.abstractmethod
    def get_data_split(self, indices: Indices) -> Any:
        pass


@dataclasses.dataclass
class ArraySplit(DataSplit):
    values: np.ndarray
    split: IndexSplit

    def get_data_split(self, indices: Indices) -> np.ndarray:
        return self.values[indices]


@dataclasses.dataclass
class ScaledArraySplit(ArraySplit):
    scaling: Union[str, None]

    def __post_init__(self):
        self.scaler = self.get_scaler()
        self.scaler.fit(self.train)  # fit only on the training data
        self.scaled_values = self.scaler.transform(self.values)

    @property
    def scaled_train(self) -> Any:
        return self.scaler.transform(self.train)

    @property
    def scaled_val(self) -> Any:
        return self.scaler.transform(self.val)

    @property
    def scaled_test(self) -> Any:
        return self.scaler.transform(self.test)

    def get_scaler(self):
        if self.scaling == None:
            return FunctionTransformer()
        elif self.scaling == 'quantile':
            n_quantiles = int(self.train.shape[0] / 2)
            return QuantileTransformer(n_quantiles=n_quantiles)
        elif self.scaling == 'standard':
            return StandardScaler()
        else:
            raise ValueError('No such scaler.')
