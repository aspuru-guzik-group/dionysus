import ml_collections
import numpy as np
from ngboost import NGBRegressor, NGBClassifier
from sklearn.model_selection import train_test_split
from .. import enums, types, utils


class NGB():

    def __init__(self, n_estimators, task, output_dim, lr, tol=1e-4):

        self.task = task
        self.output_dim = output_dim

        if self.task == enums.TaskType.regression:
            model_type = NGBRegressor
        elif self.task == enums.TaskType.binary:
            model_type = NGBClassifier
        else:
            raise NotImplementedError(f'{self.task} not implemented')

        if output_dim > 1:
            self.model = [
                model_type(n_estimators=n_estimators, random_state=utils.RANDOM_SEED, learning_rate=lr, tol=tol) for _
                in range(output_dim)
            ]  # create individual instances for each output
        else:
            self.model = [
                model_type(n_estimators=n_estimators, random_state=utils.RANDOM_SEED, learning_rate=lr, tol=tol)]

    def train(self, x, y, hp, verbose=True):
        for i in range(len(self.model)):
            self.model[i].verbose = verbose
            self.model[i].fit(x.train, y.train[:, i], x.val, y.val[:, i], early_stopping_rounds=hp.patience)

    def train_bo(self, x, y, val_size=0.15):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size)
        for i in range(len(self.model)):
            self.model[i].fit(x_train, y_train.ravel(), x_val, y_val.ravel(), early_stopping_rounds=self.hp.patience)
        # self.model.fit(x, y.ravel())

    def predict_bo(self, x):
        y_mu, y_std = [], []
        for m in self.model:
            if self.task == enums.TaskType.regression:
                y_dists = m.pred_dist(x)
                y_mu.append(y_dists.loc)
                y_std.append(np.sqrt(y_dists.var))
            elif self.task == enums.TaskType.binary:
                y_mu.append(m.predict(x))
                y_std.append(m.predict_proba(x)[:, 1])

        y_mu = np.stack(y_mu, axis=0).T
        y_std = np.stack(y_std, axis=0).T

        return y_mu, y_std

    def predict(self, smi_split: types.ArraySplit, x: types.ArraySplit, y: types.ArraySplit):
        uniq_smi = smi_split.values
        y_mu, y_std = [], []

        for m in self.model:
            if self.task == enums.TaskType.regression:
                y_dists = m.pred_dist(x.values)
                y_mu.append(y_dists.loc)
                y_std.append(np.sqrt(y_dists.var))
            elif self.task == enums.TaskType.binary:
                y_mu.append(m.predict(x.values))
                y_std.append(m.predict_proba(x.values)[:, 1])
            else:
                raise NotImplementedError(f'{self.task} not implemented')

        # transpose and turn into numpy array
        y_mu = np.transpose(y_mu).reshape(y.values.shape)
        y_std = np.transpose(y_std).reshape(y.values.shape)

        return uniq_smi, y_mu, y_std

    @classmethod
    def from_hparams(cls, hp: types.ConfigDict) -> 'NGB':
        return cls(n_estimators=hp.n_estimators,
                   task=hp.task,
                   output_dim=hp.output_dim,
                   lr=hp.lr)


def default_hp(task: enums.TaskType, output_dim: int) -> types.ConfigDict:
    hp = ml_collections.ConfigDict()
    hp.lr = 0.005
    hp.n_estimators = 2000
    hp.tol = 1e-4
    hp.task = str(enums.TaskType(task))
    # hp.output_act = modules.task_to_activation_str(hp.task)
    hp.output_dim = output_dim
    hp.patience = 100
    return hp
