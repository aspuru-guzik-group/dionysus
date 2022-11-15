import edward2 as ed
import ml_collections
import tensorflow as tf
import tensorflow.keras as tfk

from . import modules
from .. import enums
from .. import types


# import tensorflow_addons as tfa


class MLPGP(tfk.models.Model):

    def __init__(self, layer_size: int, n_layers: int, gp_inducing: int, output_dim: int,
                 output_act: types.CastableActivation):
        super(MLPGP, self).__init__()

        layers = []
        for _ in range(n_layers):
            layers.append(tfa.layers.SpectralNormalization(tfk.layers.Dense(layer_size, 'relu')))
            layers.append(tfk.layers.BatchNormalization())
        self.encoder = tfk.models.Sequential(layers)
        self.gp = ed.layers.RandomFeatureGaussianProcess(output_dim, num_inducing=gp_inducing)
        self.act = modules.cast_activation(output_act)

    def call(self, x: tf.Tensor, is_training: bool) -> tf.Tensor:
        return self.act(self.gp(self.embed(x, is_training))[0])

    def embed(self, x: tf.Tensor, is_training: bool) -> tf.Tensor:
        return self.encoder(x, training=is_training)

    @classmethod
    def from_hparams(cls, hp: types.ConfigDict) -> 'MLPGP':
        return cls(layer_size=hp.layer_size,
                   n_layers=hp.n_layers,
                   gp_inducing=hp.gp_inducing,
                   output_dim=hp.output_dim,
                   output_act=hp.output_act)


def default_hp(task: enums.TaskType, output_dim: int) -> types.ConfigDict:
    hp = ml_collections.ConfigDict()
    hp.layer_size = 100
    hp.n_layers = 3
    hp.gp_inducing = 100
    hp.task = str(enums.TaskType(task))
    hp.output_act = modules.task_to_activation_str(hp.task)
    hp.output_dim = output_dim
    hp.lr = 5e-3
    hp.epochs = 300
    hp.batch_size = 256
    return hp
