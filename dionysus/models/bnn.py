import ml_collections
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from . import modules, training
from .. import enums, types


class BNN(tf.keras.models.Model):

    def __init__(self, layer_size: int, n_layers: int, kld_beta: float, output_dim: int,
                 output_act: types.CastableActivation, task: enums.TaskType, scale_features: bool,
                 scale_targets: bool):
        super(BNN, self).__init__()

        self.scale_features = scale_features
        self.scale_targets = scale_targets
        self.task = task
        self.kld_beta = kld_beta

        self.bnn_layers = []
        for _ in range(n_layers):
            self.bnn_layers.append(tfp.layers.DenseReparameterization(layer_size, activation='relu'))
            # self.bnn_layers.append(tf.keras.layers.BatchNormalization())
        self.out_layer = tf.keras.layers.Dense(output_dim)
        self.act = modules.cast_activation(output_act)

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        for i in range(len(self.bnn_layers)):
            x = self.bnn_layers[i](x, training=training)
        x_out = self.out_layer(x)
        output = self.act(x_out)
        return output

    def train(self, x, y, hp, verbose=True):
        optimizer = tf.optimizers.Adam(hp.lr)
        main_metric, metric_fn = training.task_to_metric(self.task)

        early_stop = training.EarlyStopping(self, patience=hp.patience)
        stop_metric = f'val_{main_metric}'
        pbar = tqdm(range(hp.epochs), disable=not verbose)
        stats = []

        # scale if needed
        if self.scale_features:
            x_train, x_val, x_test = x.scaled_train, x.scaled_val, x.scaled_test
        else:
            x_train, x_val, x_test = x.train, x.val, x.test

        if self.scale_targets:
            y_train, y_val, y_test = y.scaled_train, y.scaled_val, y.scaled_test
        else:
            y_train, y_val, y_test = y.train, y.val, y.test

        # get sample weights if classification
        if self.task == enums.TaskType.binary:
            # w_train, w_val, w_test = datasets.get_sample_weights(y_train, y_val, y_test)
            w_train, w_val, w_test = None, None, None  # removed due to poor performance
        else:
            w_train, w_val, w_test = None, None, None

        # bnn uses kld regularized loss
        n = len(y_train)

        def loss_fn(y_true, y_pred, sample_weight=None):
            fn = training.task_to_loss(self.task)
            scaled_kl = tf.math.reduce_mean(self.losses) / n
            loss = fn(y_true, y_pred, sample_weight=sample_weight) + self.kld_beta * scaled_kl
            return loss

        # start training
        for _ in pbar:
            # mini-batch training
            training.train_step(self, x_train, y_train, optimizer, loss_fn, hp.batch_size, sample_weight=w_train)

            # evalulate all sets
            result = {}
            for inputs, target, weight, prefix in [
                (x_train, y_train, w_train, 'train'),
                (x_val, y_val, w_val, 'val'),
                (x_test, y_test, w_test, 'test')
            ]:
                output = self(inputs)
                result[f'{prefix}_{main_metric}'] = metric_fn(target, output, sample_weight=weight)
                result[f'{prefix}_loss'] = loss_fn(target, output, sample_weight=weight).numpy()
            stats.append(result)

            pbar.set_postfix(stats[-1])
            if early_stop.check_criteria(stats[-1][stop_metric]):
                break

        early_stop.restore_best()
        best_step = early_stop.best_step
        print(f'Early stopped at {best_step} with {stop_metric}={stats[best_step][stop_metric]:.3f}')

    def train_bo(self, x, y, val_size=0.15, verbose=True):
        # assume that hp is class attr
        optimizer = tf.optimizers.Adam(self.hp.lr)
        loss_fn = training.task_to_loss(self.task)
        main_metric, metric_fn = training.task_to_metric(self.task)

        early_stop = training.EarlyStopping(self, patience=self.hp.patience)
        stop_metric = f'val_{main_metric}'
        pbar = tqdm(range(self.hp.epochs), disable=not verbose)
        stats = []

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size)

        n = len(y_train)

        def loss_fn(y_true, y_pred, sample_weight=None):
            fn = training.task_to_loss(self.task)
            scaled_kl = tf.math.reduce_mean(self.losses) / n
            loss = fn(y_true, y_pred, sample_weight=sample_weight) + self.kld_beta * scaled_kl
            return loss

        for _ in pbar:
            training.train_step(self, x_train, y_train, optimizer, loss_fn, self.hp.batch_size)

            # for batch in training.get_batch_indices(n, self.hp.batch_size):
            #     x_batch = tf.gather(x, batch)
            #     y_batch = tf.gather(y, batch)
            #     with tf.GradientTape() as tape:
            #         output = self(x_batch)
            #         scaled_kl = tf.math.reduce_mean(self.losses) / n
            #         loss = loss_fn(y_batch, output) + scaled_kl
            #     grads = tape.gradient(loss, self.trainable_variables)
            #     optimizer.apply_gradients(zip(grads, self.trainable_variables))

            # evalulate for all
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

    def predict(self, smi_split: types.ArraySplit, x: types.ArraySplit, y: types.ArraySplit):
        uniq_smi = smi_split.values
        if self.scale_features:
            x_values = x.scaled_values
        else:
            x_values = x.values

        # predict values and the standard deviation (MC)
        predictions = []
        for _ in range(100):
            predictions.append(self(x_values).numpy())
        # predictions = np.concatenate(predictions, axis = 1)
        predictions = np.stack(predictions, axis=0)

        y_mu = np.mean(predictions, axis=0).reshape(y.values.shape)
        if self.task == enums.TaskType.regression:
            y_std = np.std(predictions, axis=0).reshape(y.values.shape)
        elif self.task == enums.TaskType.binary:
            # y_std = np.array( [1.0 - v if v < 0.5 else v for v in y_mu] )
            y_std = y_mu
            y_mu = (y_mu > 0.5).astype(int)  # covert into binary predictions

        return uniq_smi, y_mu, y_std

    def predict_bo(self, x):
        # predict values and the standard deviation (MC)
        predictions = []
        for _ in range(100):
            predictions.append(self(x).numpy())
        predictions = np.stack(predictions, axis=0)
        y_mu = np.mean(predictions, axis=0)
        if self.task == enums.TaskType.regression:
            y_std = np.std(predictions, axis=0)
        elif self.task == enums.TaskType.binary:
            y_std = y_mu
            y_mu = (y_mu > 0.5).astype(int)  # convert to binary predictions

        return y_mu, y_std

    @classmethod
    def from_hparams(cls, hp: types.ConfigDict) -> 'BNN':
        return cls(layer_size=hp.layer_size,
                   n_layers=hp.n_layers,
                   kld_beta=hp.kld_beta,
                   output_dim=hp.output_dim,
                   output_act=hp.output_act,
                   task=hp.task,
                   scale_features=hp.scale_features,
                   scale_targets=hp.scale_targets)


def default_hp(task: enums.TaskType, output_dim: int) -> types.ConfigDict:
    hp = ml_collections.ConfigDict()
    hp.layer_size = 100
    hp.n_layers = 1
    hp.task = str(enums.TaskType(task))
    hp.output_act = modules.task_to_activation_str(hp.task)
    hp.output_dim = output_dim
    hp.patience = 100
    hp.lr = 1e-3
    hp.epochs = 2000
    hp.batch_size = 256
    hp.scale_features = False
    hp.scale_targets = True
    hp.kld_beta = 0.1
    return hp
