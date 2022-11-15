import ml_collections
import official.nlp.modeling.layers as nlp_layers
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from . import modules, training
from .. import enums, types


class ResetCovarianceCallback(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        """Resets covariance matrix at the begining of the epoch."""
        if epoch > 0:
            self.model.gp.reset_covariance_matrix()


class SNGP(tf.keras.Model):

    def __init__(self, layer_size: int, n_layers: int, dropout: float,
                 output_dim: int, output_act: types.CastableActivation, task: enums.TaskType,
                 scale_features: bool, scale_targets: bool, sn_multiplier: float):
        super(SNGP, self).__init__()

        self.scale_features = scale_features
        self.scale_targets = scale_targets
        self.task = task

        self.sn_layers = []
        for _ in range(n_layers):
            self.sn_layers.append(nlp_layers.SpectralNormalization(tf.keras.layers.Dense(layer_size, 'relu'),
                                                                   norm_multiplier=sn_multiplier))
            if dropout > 0:
                self.sn_layers.append(tf.keras.layers.Dropout(dropout))

        # final GP layer
        self.gp = nlp_layers.RandomFeatureGaussianProcess(
            output_dim,
            gp_cov_momentum=-1,
            normalize_input=False,
            scale_random_features=True
        )
        self.act = modules.cast_activation(output_act)

    def embed(self, x: tf.Tensor, training: bool = False):
        for i in range(len(self.sn_layers)):
            x = self.sn_layers[i](x, training=training)
        return x

    def infer(self, x: tf.Tensor, training: bool = False):
        x_out, covmat = self.gp(x, training=training)
        output = self.act(x_out)  # activation after gp

        if self.task == enums.TaskType.regression:
            var = tf.linalg.diag_part(covmat)[:, None]
            std = tf.sqrt(var)
        elif self.task == enums.TaskType.binary:
            std = output
        else:
            raise NotImplementedError()

        return output, std

    def call(self, x: tf.Tensor, training: bool = False, return_std: bool = False) -> tf.Tensor:
        x = self.embed(x, training=training)
        output, std = self.infer(x, training=training)

        if not training and return_std:
            return output, std
        else:
            return output

    def train(self, x, y, hp, verbose: bool = True):
        optimizer = tf.optimizers.Adam(hp.lr)
        loss_fn = training.task_to_loss(self.task)
        main_metric, metric_fn = training.task_to_metric(self.task)

        early_stop = training.EarlyStopping(self, patience=hp.patience)
        stop_metric = f'val_{main_metric}'
        pbar = tqdm(range(hp.epochs), disable=not verbose)
        stats = []

        if self.scale_features:
            x_train, x_val, x_test = x.scaled_train, x.scaled_val, x.scaled_test
        else:
            x_train, x_val, x_test = x.train, x.val, x.test

        if self.scale_targets:
            y_train, y_val, y_test = y.scaled_train, y.scaled_val, y.scaled_test
        else:
            y_train, y_val, y_test = y.train, y.val, y.test

        if self.task == enums.TaskType.binary:
            # w_train, w_val, w_test = datasets.get_sample_weights(y_train, y_val, y_test)
            w_train, w_val, w_test = None, None, None  # removed due to poor performance
        else:
            w_train, w_val, w_test = None, None, None

        for _ in pbar:
            # reset the covariance every epoch for random features gp
            if _ > 0:
                self.gp.reset_covariance_matrix()

            # mini-batch training
            training.train_step(self, x_train, y_train, optimizer, loss_fn, hp.batch_size, sample_weight=w_train)

            # evaluate all sets
            result = {}
            for inputs, target, weight, prefix in [
                (x_train, y_train, w_train, 'train'),
                (x_val, y_val, w_val, 'val'),
                (x_test, y_test, w_test, 'test')
            ]:
                output = self(inputs, training=False)
                result[f'{prefix}_{main_metric}'] = metric_fn(target, output, sample_weight=weight)
                result[f'{prefix}_loss'] = loss_fn(target, output, sample_weight=weight).numpy()
            stats.append(result)

            pbar.set_postfix(stats[-1])
            if early_stop.check_criteria(stats[-1][stop_metric]):
                break

        early_stop.restore_best()
        best_step = early_stop.best_step
        print(f'Early stopped at {best_step} with {stop_metric}={stats[best_step][stop_metric]:.3f}')

    def train_bo(self, x, y, val_size=0.15, verbose: bool = True):
        # assumes hp is set as class attr
        optimizer = tf.optimizers.Adam(self.hp.lr)
        loss_fn = training.task_to_loss(self.task)
        main_metric, metric_fn = training.task_to_metric(self.task)

        early_stop = training.EarlyStopping(self, patience=self.hp.patience)
        stop_metric = f'val_{main_metric}'
        pbar = tqdm(range(self.hp.epochs), disable=not verbose)
        stats = []

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size)

        for _ in pbar:
            # reset the covariance every epoch for random features gp
            if _ > 0:
                self.gp.reset_covariance_matrix()

            training.train_step(self, x_train, y_train, optimizer, loss_fn, self.hp.batch_size)

            result = {}
            for inputs, target, prefix in [
                (x_train, y_train, 'train'),
                (x_val, y_val, 'val')
            ]:
                output = self(inputs)
                result[f'{prefix}_{main_metric}'] = metric_fn(target, output)
                result[f'{prefix}_loss'] = loss_fn(target, output).numpy()
            stats.append(result)

            pbar.set_postfix(stats[-1])
            if early_stop.check_criteria(stats[-1][stop_metric]):
                break

        early_stop.restore_best()
        best_step = early_stop.best_step
        print(f'Early stopped at {best_step} with {stop_metric}={stats[best_step][stop_metric]:.3f}')

    def predict(self, smi_split: types.ArraySplit, x: types.ArraySplit, y: types.ArraySplit,
                return_embeddings: bool = False):

        uniq_smi = smi_split.values
        if self.scale_features:
            x_values = x.scaled_values
        else:
            x_values = x.values

        # predict values and the standard deviation
        y_mu, y_std = self.call(x_values, training=False, return_std=True)
        if self.task == enums.TaskType.regression:
            y_mu = y_mu.numpy()
            y_std = y_std.numpy()
        elif self.task == enums.TaskType.binary:
            # y_std = np.array( [1.0 - v if v < 0.5 else v for v in y_mu.numpy()] )
            # y_std = y_mu.numpy()
            y_mu = (y_mu.numpy() > 0.5).astype(int)  # prediction for classification is based on probability threshold
            y_std = y_std.numpy()
        else:
            raise NotImplementedError()

        if return_embeddings:
            embeddings = self.embed(x_values).numpy()
            return uniq_smi, y_mu, y_std, embeddings

        return uniq_smi, y_mu, y_std

    def predict_bo(self, x):
        # predict values and the standard deviation
        y_mu, y_std = self.call(x, training=False, return_std=True)
        if self.task == enums.TaskType.regression:
            y_mu = y_mu.numpy()
            y_std = y_std.numpy()
        elif self.task == enums.TaskType.binary:
            y_mu = (y_mu.numpy() > 0.5).astype(int)  # prediction for classification is based on probability threshold
            y_std = y_std.numpy()
        else:
            raise NotImplementedError()

        return y_mu, y_std

    @classmethod
    def from_hparams(cls, hp: types.ConfigDict) -> 'SNGP':
        return cls(
            layer_size=hp.layer_size,
            n_layers=hp.n_layers,
            dropout=hp.dropout,
            sn_multiplier=hp.sn_multiplier,
            output_dim=hp.output_dim,
            output_act=hp.output_act,
            task=hp.task,
            scale_features=hp.scale_features,
            scale_targets=hp.scale_targets,
        )


def default_hp(task: enums.TaskType, output_dim: int) -> types.ConfigDict:
    hp = ml_collections.ConfigDict()
    hp.layer_size = 100
    hp.n_layers = 3
    hp.sn_multiplier = 0.9
    hp.dropout = 0  # 0.05
    hp.task = str(enums.TaskType(task))
    hp.output_act = modules.task_to_activation_str(hp.task)
    hp.output_dim = output_dim
    hp.patience = 200
    hp.lr = 1e-3
    hp.epochs = 2000
    hp.batch_size = 256
    hp.scale_features = False
    hp.scale_targets = True
    return hp
