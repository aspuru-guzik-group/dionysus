import ml_collections
import sonnet as snt
import tensorflow as tf

from . import modules
from .. import enums
from .. import types


class MLP(snt.Module):

    def __init__(self, layer_size: int, n_layers: int, dropout: float, output_dim: int,
                 output_act=types.CastableActivation, name='MLP'):
        super(MLP, self).__init__(name=name)
        self.dropout = dropout
        self.layers = [snt.Linear(layer_size) for _ in range(n_layers)]
        self.normalizers = [snt.BatchNorm(True, True) for _ in range(n_layers)]
        self.act = modules.cast_activation('relu')
        self.pred = modules.get_pred_layer(output_dim, output_act)

    def __call__(self, x: tf.Tensor, is_training: bool) -> tf.Tensor:
        return self.pred(self.encode(x, is_training))

    def encode(self, x: tf.Tensor, is_training: bool) -> tf.Tensor:
        use_dropout = self.dropout > 0
        for i, (layer, norm) in enumerate(zip(self.layers, self.normalizers)):
            x = layer(x)
            if use_dropout and is_training:
                x = tf.nn.dropout(x, rate=self.dropout)
            x = self.act(x)
            x = norm(x, is_training)
        return x

    def embed(self, x: tf.Tensor) -> tf.Tensor:
        return self.encode(x, False)

    @classmethod
    def from_hparams(cls, hp: types.ConfigDict) -> 'MLP':
        return cls(layer_size=hp.layer_size,
                   n_layers=hp.n_layers,
                   dropout=hp.dropout,
                   output_dim=hp.output_dim,
                   output_act=hp.output_act)


def default_hp(task: enums.TaskType, output_dim: int) -> types.ConfigDict:
    hp = ml_collections.ConfigDict()
    hp.layer_size = 100
    hp.n_layers = 3
    hp.dropout = 0.05
    hp.task = str(enums.TaskType(task))
    hp.output_act = modules.task_to_activation_str(hp.task)
    hp.output_dim = output_dim
    hp.lr = 1e-3
    hp.epochs = 2000
    hp.batch_size = 32
    return hp
