import ml_collections
import official.nlp.modeling.layers as nlp_layers
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

from . import modules, training
from .gnn import GNN
from .. import enums, types, graphs



class GNNGP(GNN):
    ''' GNN with gaussian process as a predicing layer
    '''

    def __init__(self,
                 node_size: int,
                 edge_size: int,
                 global_size: int,
                 output_dim: int,
                 task: enums.TaskType,
                 block_type: enums.GNNBlock,
                 output_act: types.CastableActivation,
                 n_layers: int,
                 scale_targets: bool):
        super(GNNGP, self).__init__(node_size, edge_size, global_size,
                                    output_dim, block_type, output_act, n_layers)

        self.scale_targets = scale_targets
        self.task = task

        self.gp = nlp_layers.RandomFeatureGaussianProcess(
            output_dim,
            gp_cov_momentum=-1,
            normalize_input=False,
            scale_random_features=True
        )
        self.act = modules.cast_activation(output_act)

    def __call__(self, x: tf.Tensor, training: bool = True, return_std: bool = False):
        output, covmat = self.gp(super(GNNGP, self).embed(x), training=training)
        output = self.act(output)

        if self.task == enums.TaskType.regression:
            var = tf.linalg.diag_part(covmat)[:, None]
            std = tf.sqrt(var)
        elif self.task == enums.TaskType.binary:
            # probability given by output
            std = output
        else:
            raise NotImplementedError()

        if not training and return_std:
            return output, std
        else:
            return output

    def train(self, x, y, hp, verbose=True):
        assert isinstance(x.train, types.GraphsTuple), 'GNN requires graph tuples input.'
        optimizer = tf.optimizers.Adam(hp.lr)
        loss_fn = training.task_to_loss(self.task)
        main_metric, metric_fn = training.task_to_metric(self.task)

        early_stop = training.EarlyStopping(self, patience=hp.patience)
        stop_metric = f'val_{main_metric}'
        pbar = tqdm(range(hp.epochs), disable=not verbose)
        stats = []

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
            if _ > 0:
                self.gp.reset_covariance_matrix()

            # mini-batch training
            training.train_step(self, x.train, y_train, optimizer, loss_fn, hp.batch_size, sample_weight=w_train)

            # evaluate all sets
            result = {}
            for inputs, target, weight, prefix in [
                (x.train, y_train, w_train, 'train'),
                (x.val, y_val, w_val, 'val'),
                (x.test, y_test, w_test, 'test')
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
        # assume that the hp has been set as class attribute
        # assert isinstance(x.train, types.GraphsTuple), 'GNN requires graph tuples input.'
        optimizer = tf.optimizers.Adam(self.hp.lr)
        loss_fn = training.task_to_loss(self.task)
        main_metric, metric_fn = training.task_to_metric(self.task)

        early_stop = training.EarlyStopping(self, patience=self.hp.patience)
        stop_metric = f'val_{main_metric}'
        pbar = tqdm(range(self.hp.epochs), disable=not verbose)
        stats = []

        t_idx, v_idx, y_train, y_val = train_test_split(range(len(y)), y, test_size=val_size)
        x_train = graphs.get_graphs(x, np.array(t_idx))
        x_val = graphs.get_graphs(x, np.array(v_idx))

        for _ in pbar:
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
        assert isinstance(x.train, types.GraphsTuple), 'GNN requires graph tuples input.'
        uniq_smi = smi_split.values

        # predict values and the standard deviation
        y_mu, y_std = self.__call__(x.values, training=False, return_std=True)

        if self.task == enums.TaskType.regression:
            y_mu = y_mu.numpy()
            y_std = y_std.numpy()
        elif self.task == enums.TaskType.binary:
            y_std = y_mu.numpy()
            y_mu = (y_mu.numpy() > 0.5).astype(int)  # output prediction based on probability
        else:
            raise NotImplementedError()

        if return_embeddings:
            embeddings = self.embed(x_values).numpy()
            return uniq_smi, y_mu, y_std, embeddings

        return uniq_smi, y_mu, y_std

    def predict_bo(self, x):
        # predict values and the standard deviation
        y_mu, y_std = self.__call__(x, training=False, return_std=True)

        if self.task == enums.TaskType.regression:
            y_mu = y_mu.numpy()
            y_std = y_std.numpy()
        elif self.task == enums.TaskType.binary:
            y_std = y_mu.numpy()
            y_mu = (y_mu.numpy() > 0.5).astype(int)  # output prediction based on probability
        else:
            raise NotImplementedError()

        return y_mu, y_std

    @classmethod
    def from_hparams(cls, hp: types.ConfigDict) -> 'GNNGP':
        return cls(node_size=hp.node_size,
                   edge_size=hp.edge_size,
                   global_size=hp.global_size,
                   output_dim=hp.output_dim,
                   task=hp.task,
                   block_type=enums.GNNBlock(hp.block_type),
                   output_act=hp.output_act,
                   n_layers=hp.n_layers,
                   scale_targets=hp.scale_targets)


def default_hp(task: enums.TaskType, output_dim: int) -> types.ConfigDict:
    hp = ml_collections.ConfigDict()
    hp.node_size = 50
    hp.edge_size = 20
    hp.global_size = 150
    hp.block_type = 'graphnet'
    hp.n_layers = 3
    hp.task = str(enums.TaskType(task))
    hp.output_act = modules.task_to_activation_str(hp.task)
    hp.output_dim = output_dim
    hp.patience = 100
    hp.lr = 1e-3
    hp.epochs = 2000
    hp.batch_size = 256
    hp.scale_targets = True
    return hp
