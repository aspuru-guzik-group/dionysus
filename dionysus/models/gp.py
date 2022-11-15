import gpflow
import ml_collections
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from . import modules, training
from .. import enums, types


class GP():

    def __init__(self, kernel: str, input_dim: int, output_dim: int, task: enums.TaskType, scale_features: bool,
                 scale_targets: bool):

        self.scale_features = scale_features
        self.scale_targets = scale_targets
        self.task = task
        self.output_dim = output_dim
        self.input_dim = input_dim

        if kernel == 'tanimoto':
            self.kernel = Tanimoto()
        elif kernel == 'rbf':
            # use high dimensional kernel
            self.kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0 for _ in range(input_dim)])
        else:
            raise NotImplementedError('No such kernel for GP.')

        if self.task == enums.TaskType.regression:
            self.model = gpflow.models.GPR
        elif self.task == enums.TaskType.binary:
            self.model = gpflow.models.VGP
            self.likelihood = gpflow.likelihoods.Bernoulli()
        else:
            raise NotImplementedError(f'{self.task} not implemented.')

    def train(self, x, y, hp, verbose=True):
        optimizer = tf.optimizers.Adam(hp.lr)
        loss_fn = training.task_to_loss(self.task)
        main_metric, metric_fn = training.task_to_metric(self.task)

        if self.scale_features:
            x_train, x_val, x_test = x.scaled_train, x.scaled_val, x.scaled_test
        else:
            x_train, x_val, x_test = x.train, x.val, x.test

        if self.scale_targets:
            y_train, y_val, y_test = y.scaled_train, y.scaled_val, y.scaled_test
        else:
            y_train, y_val, y_test = y.train, y.val, y.test

        # create the model (gpflow requires the data to create model)
        x_train, x_val, x_test = x_train.astype(np.float64), x_val.astype(np.float64), x_test.astype(np.float64)
        if self.task == enums.TaskType.regression:
            y_train, y_val, y_test = y_train.astype(np.float64), y_val.astype(np.float64), y_test.astype(np.float64)
            self.gp = self.model(data=(x_train, y_train), kernel=self.kernel)
        elif self.task == enums.TaskType.binary:
            self.gp = self.model(data=(x_train, y_train), likelihood=self.likelihood, kernel=self.kernel)
        else:
            raise NotImplementedError(f'{self.task} not implemented.')

        if tf.math.is_nan(self.gp.training_loss()):
            print(f'Error in kernel {self.kernel}, defaulting to RBF.')
            self.gp = self.model(
                data=(x_train, y_train),
                kernel=gpflow.kernels.SquaredExponential(lengthscales=[1.0 for _ in range(input_dim)])
            )

        early_stop = training.EarlyStopping(self.gp, patience=hp.patience)
        stop_metric = f'val_{main_metric}'
        pbar = tqdm(range(hp.epochs), disable=not verbose)
        stats = []

        for _ in pbar:
            # training
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.gp.trainable_variables)
                loss = self.gp.training_loss()
            grads = tape.gradient(loss, self.gp.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.gp.trainable_variables))

            # evalulate for all sets
            result = {}
            for inputs, target, prefix in [
                (x_train, y_train, 'train'),
                (x_val, y_val, 'val'),
                (x_test, y_test, 'test')
            ]:
                y_pred, _ = self.gp.predict_y(inputs)
                result[f'{prefix}_{main_metric}'] = metric_fn(target, y_pred)
                result[f'{prefix}_loss'] = np.mean(loss_fn(target, y_pred).numpy())

            stats.append(result)
            pbar.set_postfix(stats[-1])
            if early_stop.check_criteria(stats[-1][stop_metric]):
                break

        early_stop.restore_best()
        best_step = early_stop.best_step
        print(f'Early stopped at {best_step} with {stop_metric}={stats[best_step][stop_metric]:.3f}')

    def train_bo(self, x, y, val_size=0.15, verbose=True):
        # hp must be here set as a class attr

        optimizer = tf.optimizers.Adam(self.hp.lr)
        loss_fn = training.task_to_loss(self.task)
        main_metric, metric_fn = training.task_to_metric(self.task)

        x = x.astype(np.float64)
        y = y.astype(np.float64)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size)

        # change to double
        if self.task == enums.TaskType.regression:
            # x, y = x.astype(np.float64), y.astype(np.float64)
            self.gp = self.model(data=(x_train, y_train), kernel=self.kernel)
        elif self.task == enums.TaskType.binary:
            self.gp = self.model(data=(x_train, y_train), likelihood=self.likelihood, kernel=self.kernel)
        else:
            raise NotImplementedError(f'{self.task} not implemented.')

        if tf.math.is_nan(self.gp.training_loss()):
            print(f'Error in kernel {self.kernel}, defaulting to RBF.')
            self.gp = self.model(data=(x_train, y_train), kernel=gpflow.kernels.SquaredExponential())

        # cant do early stopping (no validation set)
        early_stop = training.EarlyStopping(self.gp, patience=self.hp.patience)
        stop_metric = f'val_{main_metric}'
        pbar = tqdm(range(self.hp.epochs), disable=not verbose)
        stats = []

        for _ in pbar:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.gp.trainable_variables)
                loss = self.gp.training_loss()
            grads = tape.gradient(loss, self.gp.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.gp.trainable_variables))

            # evalulate results for training set
            result = {}
            for inputs, target, prefix in [
                (x_train, y_train, 'train'),
                (x_val, y_val, 'val')
            ]:
                output, _ = self.gp.predict_y(inputs)
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
            x_values = x.scaled_values.astype(np.float64)
        else:
            x_values = x.values.astype(np.float64)

        y_mu, y_var = self.gp.predict_y(x_values)
        if self.task == enums.TaskType.regression:
            y_mu = y_mu.numpy()
            y_std = np.sqrt(y_var.numpy())
        elif self.task == enums.TaskType.binary:
            # y_std = np.array( [1.0 - v if v < 0.5 else v for v in y_mu.numpy()] )
            # y_std = y_var.numpy()
            y_std = y_mu.numpy()
            y_mu = (y_mu.numpy() > 0.5).astype(int)
        else:
            raise NotImplementedError(f'{self.task} not implemented.')

        return uniq_smi, y_mu, y_std

    def predict_bo(self, x):
        x = x.astype(np.float64)
        y_mu, y_var = self.gp.predict_y(x)
        if self.task == enums.TaskType.regression:
            y_mu = y_mu.numpy()
            y_std = np.sqrt(y_var.numpy())
        elif self.task == enums.TaskType.binary:
            y_std = y_mu.numpy()
            y_mu = (y_mu.numpy() > 0.5).astype(int)
        return y_mu, y_std

    def save(self, filepath):
        tf.saved_model.save(self.gp, filepath)

    def load(self, filepath):
        self.gp = tf.saved_model.load(filepath)

    @classmethod
    def from_hparams(cls, hp: types.ConfigDict) -> 'GP':
        return cls(kernel=hp.kernel,
                   input_dim=hp.input_dim,
                   output_dim=hp.output_dim,
                   task=hp.task,
                   scale_features=hp.scale_features,
                   scale_targets=hp.scale_targets)


def default_hp(task: enums.TaskType, input_dim: int, output_dim: int) -> types.ConfigDict:
    hp = ml_collections.ConfigDict()
    hp.kernel = 'rbf'
    hp.task = str(enums.TaskType(task))
    hp.output_act = modules.task_to_activation_str(hp.task)
    hp.input_dim = input_dim
    hp.output_dim = output_dim
    hp.lr = 1e-2
    hp.epochs = 2000
    hp.scale_features = False
    hp.scale_targets = True
    hp.patience = 400
    return hp


###################################################################################
# Custom Tanimoto kernel

class Tanimoto(gpflow.kernels.Kernel):
    ''' Tanimoto kernel for gpflow from FlowMO (https://arxiv.org/abs/2010.01118)
    '''

    def __init__(self):
        super().__init__()
        # We constrain the value of the kernel variance to be positive when it's being optimised
        self.variance = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())

    def K(self, X, X2=None):
        """
        Compute the Tanimoto kernel matrix simga^2 * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))
        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        outer_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula

        denominator = -outer_product + gpflow.utilities.ops.broadcasting_elementwise(tf.add, Xs, X2s)

        return self.variance * outer_product / denominator

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
