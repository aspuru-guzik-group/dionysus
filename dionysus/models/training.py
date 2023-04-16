"""Functions related to a training, including utility for GNNs."""
import tempfile
from typing import Tuple
from warnings import warn

import numpy as np
import sklearn.metrics
import sonnet as snt
import tensorflow as tf

from .. import enums, types, graphs


def task_to_loss(task: enums.TaskType) -> types.LossFunction:
    """Map tasktype to activation."""
    task = enums.TaskType(task)
    loss_map = {
        # enums.TaskType.regression: tf.losses.mean_squared_error,
        # enums.TaskType.binary: tf.losses.binary_crossentropy,
        enums.TaskType.regression: tf.keras.losses.MeanSquaredError(),
        enums.TaskType.binary: tf.keras.losses.BinaryCrossentropy()
    }
    return loss_map[task]


def task_to_metric(task: enums.TaskType) -> Tuple[str, types.MetricFunction]:
    """Map tasktype to activation."""
    task = enums.TaskType(task)
    metric_map = {
        # enums.TaskType.regression: ('mse', tf.keras.losses.MeanSquaredError()),
        # enums.TaskType.binary: ('bce', tf.keras.losses.BinaryCrossentropy())
        enums.TaskType.regression: ('R^2', sklearn.metrics.r2_score),
        enums.TaskType.binary: ('auroc', roc_auc_score)
        # accuracy score can only take in binary values
        # enums.TaskType.binary: ('acc', lambda x, y: sklearn.metrics.accuracy_score(x, np.array(y > 0.5, dtype=int))),
    }
    return metric_map[task]

def roc_auc_score(y_true, y_pred):
    try: 
        score = sklearn.metrics.roc_auc_score(y_true, y_pred)
    except:
        # if only one class is present, then we artificially add
        # a point with different class
        if y_true.shape[-1] == 1:
            if len(np.unique(y_true)) == 1:     # only one class found
                possible_classes = np.array([0, 1])
                new_p = np.delete(possible_classes, np.unique(y_true)).reshape(-1, 1)
                y_true = np.append(y_true, new_p, axis=0)
                y_pred = np.append(y_pred, np.array([[0.5]]), axis=0)
                score = sklearn.metrics.roc_auc_score(y_true, y_pred)
            else:
                print('ROC failed, and more than one class found.')
        else:
            print('More than one dimension found.')
    return score


def get_batch_indices(n: int, batch_size: int) -> np.ndarray:
    """Gets shuffled constant size batch indices to train a model."""
    n_batches = n // batch_size
    indices = tf.random.shuffle(tf.range(n))
    indices = indices[:n_batches * batch_size]
    indices = tf.reshape(indices, (n_batches, batch_size))
    return indices


class EarlyStopping():
    """Stop training early if a metric stops improving.

      Implementation is based on keras's callback but for sonnet.
      Models often benefit from stoping early after a metric stops improving.
      This implementation assumes the monitored value will be loss-like
      (i.g. val_loss) and will checkpoint when reaching a new best value.
      Checkpointed value can be restored.

      Args:
        model: sonnet model to checkpoint.
        patience: number of iterations before flaggin a stop.
        min_delta: minimum value to quanlify as an improvement.
        checkpoint_interval: number of iterations before checkpointing.
        mode: maximise or minimise the monitor value
    """

    def __init__(self,
                 model: snt.Module,
                 patience: int = 100,
                 min_delta: float = 1e-3,
                 checkpoint_interval: int = 1,
                 mode: bool = 'maximize'):
        self.patience = patience
        self.min_delta = np.abs(min_delta)
        self.wait = 0
        self.best_step = 0
        self.checkpoint_count = 0
        self.checkpoint_interval = checkpoint_interval
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.values = []
        self.checkpoint = tf.train.Checkpoint(model=model)
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.tmp_dir.name, max_to_keep=5)
        if mode == 'maximize':
            self.monitor_op = lambda a, b: np.greater(a - min_delta, b)
            self.best_value = -np.inf
        elif mode == 'minimize':
            self.monitor_op = lambda a, b: np.less(a + min_delta, b)
            self.best_value = np.inf
        else:
            raise ValueError('Invalid mode for early stopping.')

    def check_criteria(self, monitor_value: float) -> bool:
        """Gets learing rate based on value to monitor."""
        self.values.append(monitor_value)
        self.checkpoint_count += 1

        if self.monitor_op(monitor_value, self.best_value):
            self.best_value = monitor_value
            self.best_step = len(self.values) - 1
            self.wait = 0
            if self.checkpoint_count >= self.checkpoint_interval:
                self.manager.save()
                self.checkpoint_count = 0
        else:
            self.wait += 1

        return self.wait >= self.patience

    def restore_best(self):
        self.checkpoint.restore(self.manager.latest_checkpoint)
        print(f'Restoring checkpoint at step {self.best_step} with best value at {self.best_value:.3f}')


def train_step(model, x_train, y_train, optimizer, loss_fn, batch_size, sample_weight=None):
    ''' Do a single training step and train. Weight the loss function if 
    sample_weight provided.
    '''
    n = len(y_train)
    fn = graphs.get_graphs if isinstance(x_train, types.GraphsTuple) else tf.gather
    for batch in get_batch_indices(n, batch_size):
        x_batch = fn(x_train, batch)
        y_batch = tf.gather(y_train, batch)
        w_batch = tf.gather(sample_weight, batch) if sample_weight is not None else sample_weight
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = loss_fn(y_batch, y_pred, sample_weight=w_batch)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


def evaluate_step(model, training, validation, testing, loss_fn, main_metric, metric_fn):
    ''' DEPRECATED. Evaluate the training, val and testing sets.
    '''
    warn('DEPRECATED: loop through sets and evaluate to avoid cumbersome import statements.',
         DeprecationWarning, stacklevel=2)
    result = {}
    training, validation, testing = list(training), list(validation), list(testing)
    training.append('train')
    validation.append('val')
    testing.append('test')

    for inputs, target, prefix in [tuple(training), tuple(validation), tuple(testing)]:
        y_pred = model(inputs)
        result[f'{prefix}_{main_metric}'] = metric_fn(target, y_pred)
        result[f'{prefix}_loss'] = loss_fn(target, y_pred).numpy()
    return result
