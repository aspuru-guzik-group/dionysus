#!/usr/bin/env python
import os, sys
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from graph_nets import utils_tf

from sklearn.preprocessing import MinMaxScaler

from .bo_utils import AcquisitionFunction, predict_test_set, get_scalers

from .. import utils, files, models, enums, datasets, chem
from ..models import BNN, GP, NGB, GNNGP, SNGP
from ..graphs import get_graphs


class SynthBO():

    """ Main class for synthetic Bayesian optimization and active learning experiments 
    using the DIONYSUS datasets and models. 

    Args: 
        dataset_name (str): the name of the DIONYSUS dataset
        goal (str): optimization goal, 'maximize' or 'minimize'. For binary classification
            problems, 'maximize' corresponds to searching for positive (1) labels, while
            'minimize' is analogous to searching for negative (0) labels
        model_type (str): DIONYSUS surrogate model name
        feature_type (str): featurization method for the molecules 
        acq_func_type (str): acquisition function type 
        beta (float): mean-std tradeoff parameter for the UCB acquisition function
        num_acq_samples (int): number of samples drawn in each round of acquisition function
            optimization
        batch_size (int): number of recommendations provided by the acquisition function at each 
            iteration 
        budget (int): maximum tolerated objective function measurements for a single 
            optimization campaign
        init_design_strategy (str): strategy used to select the initial design points
        num_init_design (int): number of initial design points to propose
    """

    def __init__(
        self,
        dataset_name,
        goal,
        model_type,
        feature_type,
        acq_func_type,
        beta=0.2,
        num_acq_samples=50,
        batch_size=1,
        budget=15,
        init_design_strategy='random',
        num_init_design=0.05,
        work_dir = '.',
        *args, 
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.goal = goal
        self.model_type = model_type
        self.feature_type = feature_type
        self.acq_func_type = acq_func_type
        self.beta = beta
        self.num_acq_samples = num_acq_samples
        self.batch_size = batch_size
        self.budget = budget
        self.init_design_strategy = init_design_strategy
        self.work_dir = work_dir

        # always use mfp if using baselines
        if model_type in ['random', 'nn']:
            self.feature_type = enums.FeatureType.mfp

        # process the dataset
        self._process_dataset()

        # if give integer, use this number
        if isinstance(num_init_design, int):
            self.num_init_design = num_init_design
        elif isinstance(num_init_design, float):
            assert 0. <=  num_init_design <= 1.
            self.num_init_design = int(num_init_design*self.df.shape[0])
            if self.num_init_design < 25:
                self.num_init_design = 25       # minimum allowed is 25
            elif self.num_init_design > 100:
                self.num_init_design = 100

        # print('num_init_design : ', self.num_init_design)
        # print('budget : ', self.budget)

        # quit()


        # check budget
        self._validate_budget()

        # initialize the surrogate model
        self._initialize_model(self.task_type)


        # register acquisition function
        self.acq_func = AcquisitionFunction(
            acq_func_type=self.acq_func_type, task_type=str(self.task_type), beta=self.beta
        )

    def _initialize_model(self, task):
        ''' initialize the surrogate model 

        Args: 
            task (enums.TaskType): the task type, currently we support either regression or
                binary tasks
        '''
        # initalize model
        if self.model_type == 'ngboost':
            hp = models.ngb.default_hp(task, output_dim=1)
            hp.n_estimators = 1000
            self.model = NGB.from_hparams(hp)
            self.model.hp = hp
        elif self.model_type == 'gp':
            hp = models.gp.default_hp(task, input_dim=self.input_dim, output_dim=self.output_dim)
            hp.kernel='tanimoto' if self.feature_set == enums.FeatureType.mfp else 'rbf'
            hp.epochs=1000 # fix the number of epochs
            hp.patience=200 # fix early stopping patience
            self.model = GP.from_hparams(hp)
            self.model.hp = hp # set hp as attr
        elif self.model_type == 'bnn':
            hp = models.bnn.default_hp(task, output_dim=1)
            hp.epochs=1000 # fix the number of epochs
            hp.kld_beta=0.01
            # hp.patience=100 # fix early stopping patience
            self.model = BNN.from_hparams(hp)
            self.model.hp = hp
        elif self.model_type == 'gnngp':
            hp = models.gnngp.default_hp(task, output_dim=1)
            # set custom hparams
            hp.epochs=1000 # fix the number of epochs
            self.model = GNNGP.from_hparams(hp)
            self.model.hp = hp
        elif self.model_type == 'sngp':
            hp = models.sngp.default_hp(task, output_dim=1)
            hp.epochs=1000 # fix the number of epochs
            self.model = SNGP.from_hparams(hp)
            self.model.hp = hp
        elif self.model_type == 'random':
            self.model = None
        elif self.model_type == 'nn':
            self.model = None
        else:
            raise NotImplementedError

    def _validate_budget(self):
        ''' validate that the budget does not exceed the total number of
        options in the dataset
        '''
        if self.budget > self.num_total:
            print(f'Requested budget exceeeds the number of total candidates. Resetting to {self.num_total}')
            self.budget = self.num_total - self.num_init_design

    def _validate_bo_parameters(self, dataset_config):
        ''' validate that the requested acquisition function is compatible 
        with the dataset

        Args: 

            dataset_config (obj): dataset config object
        '''
        # assert len(dataset_config.tasks) == 1
        self.task_type = dataset_config.tasks[0]
        if self.task_type == enums.TaskType.regression:
            assert self.acq_func_type in ['ei', 'ucb', 'lcb', 'variance_sampling']
        elif self.task_type == enums.TaskType.binary:
            if len(dataset_config.tasks)==1:
                assert self.acq_func_type in ['greedy_binary']
            elif len(dataset_config.tasks)>1:
                assert self.acq_func_type == 'greedy_multi_label'
        else:
            raise NotImplementedError


    def _process_dataset(self):
        ''' Process dataset object into pandas DataFrame used for 
        the sequential learning experiments. Generates dataframe containing the
        molecule smiles, feature representation and target values
        '''
        # check the dataset and feature type
        self.dataset = enums.Dataset[self.dataset_name]
        self.feature_set = enums.FeatureType[self.feature_type]
        if self.model_type in ['random', 'nn']:
            model = enums.Models['gp'] # dummy
        else:
            model = enums.Models[self.model_type] 
        #task = enums.TaskType.regression

        # load the dataset
        _, self.config, _ = datasets.load_dataset(self.dataset, self.work_dir)

        if all(t==enums.TaskType.regression for t in self.config.tasks):
            split_type = 'random'
        else:
            split_type = 'stratified'

        # check if the provided bo parameters match with the dataset task type
        self._validate_bo_parameters(self.config)

        # load task
        smi_split, X, y = datasets.load_task(
            self.dataset, self.feature_set, model, self.work_dir, task_index=0,
        )

        if self.feature_type == 'graphnet':
            train_ix = X.split.train
            val_ix = X.split.val
            test_ix = X.split.test
            work_ix = np.concatenate((train_ix, val_ix))

            # heldout test set, only used for predictions of the surrogate model
            # val set is ~ 20% of the total data
            self.test_smi_list = smi_split.test.tolist()
            self.test_X = X.test
            self.test_y = y.test

            # work set, available to the acquisition function to query
            # train set and test set together are ~80% of the dataset
            self.smi_list = smi_split.train.tolist() + smi_split.val.tolist() # smiles
            self.X = X.get_data_split(work_ix) # graphs
            self.y = np.concatenate((y.train, y.val)) # objective values

        else:

            # heldout test set, only used for predictions of the surrogate model
            # val set is ~ 20% of the total data
            self.test_smi_list = smi_split.test.tolist()
            self.test_X = X.test
            self.test_y = y.test

            # work set, available to the acquisition function to query
            # train set and test set together are ~80% of the dataset

            self.smi_list = smi_split.train.tolist() + smi_split.val.tolist() # smiles
            self.X = np.concatenate((X.train, X.val)) # feature vectors
            self.y = np.concatenate((y.train, y.val)) # objective values

            print(len(self.smi_list))
            print(len(self.test_smi_list))

        self.output_dim = self.y.shape[-1]
        #assert output_dim == 1 # only do scalar objective BayesOpt here
        self.num_total = self.y.shape[0]  # num total molecules

        # process features
        if self.feature_type == 'graphnet':
            self.input_dim = None
            X_df = [get_graphs(self.X, np.array([i])) for i in range(self.y.shape[0])]
        else:
            self.input_dim = self.X.shape[-1]
            X_df = [[X_elem] for X_elem in self.X]

        self.df = pd.DataFrame({'smi': self.smi_list, 'x': X_df, 'y': self.y.flatten()})


    def run(self, num_restarts, eval_metrics = False): 
        ''' Run the sequential learning experiments for num_restart independently seeded
        executions. The results for each run are stored in a list of pandas DataFrames and 
        saved in the pickle file res_file

        Args: 
            num_restarts (int): number of independent excecutions 
            eval_metrics (bool): toggles evaluating the surrogate on a held-out set (set to False
                if not looking at the performance/calibration, will save inference time)
        '''

        df_optimization = []            # collects optimization trace
        df_metrics = []                 # collects metrics during optimization (if specified)

        for num_restart in range(num_restarts):

            keep_running = True
            while keep_running:
                # try: 
                observations = []  # [ {'smiles': , 'y': }, ..., ]
                all_test_set_metrics = []

                iter_num = 0

                while (len(observations)-self.num_init_design) < self.budget:

                    # re-initialize the surrogate model
                    self._initialize_model(self.task_type)

                    meas_df, avail_df = self.split_avail(self.df, observations)
                    # shuffle the available candidates (acq func sampling)
                    avail_df = avail_df.sample(frac=1).reset_index(drop=True)

                    print(f'RESTART : {num_restart+1}\tNUM_ITER : {iter_num}\tNUM OBS : {len(observations)}')

                    if self.model_type == 'random':
                        # always use init design strategy
                        is_init_design = True
                    else:
                        is_init_design = len(observations) < self.num_init_design

                    if is_init_design:
                        # sample randomly
                        sample, measurement = self.sample_meas_randomly(avail_df)
                        observations.append({'smi': sample, 'y': measurement})
                    elif self.model_type == 'nn':
                        # use nearnest neighbour strategy
                        X_avail, _ = self.make_xy(avail_df, num=self.num_acq_samples)
                        if self.goal == 'minimize':
                            best_fp = np.array(meas_df.nsmallest(n=1, columns='y')['x'].tolist()).squeeze()
                        elif self.goal == 'maximize':
                            best_fp = np.array(meas_df.nlargest(n=1, columns='y')['x'].tolist()).squeeze()
                        sims = np.array([chem.similarity_between_fps(x, best_fp) for x in X_avail])
                        ind = np.argsort(sims)[-self.batch_size:]
                        for sample_idx in ind:
                            sample, measurement = self.sample_meas_acq(avail_df, sample_idx)
                        observations.append({'smi': sample, 'y': measurement})
                    else:
                        # sample using surrogate model and acquisition
                        X_meas, y_meas = self.make_xy(meas_df)
                        X_avail, _ = self.make_xy(avail_df, num=self.num_acq_samples)

                        # get the scalers
                        X_scaler, y_scaler = get_scalers(
                            self.feature_type, self.model_type, self.task_type
                        )

                        # fit the scalers
                        X_scaler = X_scaler.fit(X_meas)
                        y_scaler = y_scaler.fit(y_meas)

                        # transform the current measurements
                        X_meas_scal = X_scaler.transform(X_meas)                    # (num_obs, X_dim)
                        y_meas_scal = y_scaler.transform(y_meas)                    # (num_obs, 1)

                        # transform the features of the available candidates
                        X_avail_scal = X_scaler.transform(X_avail)                  # (num_acq_samples, 1)

                        # get the scaled incumbent point
                        if self.goal == 'minimize':
                            incumbent_scal = np.amin(y_meas_scal)
                        elif self.goal == 'maximize':
                            incumbent_scal = np.amax(y_meas_scal)

                        # train the model on observations
                        self.model.train_bo(X_meas_scal, y_meas_scal)                   # (num_acq_samples, 1)

                        if self.task_type == enums.TaskType.regression:
                            mu_avail, sigma_avail = self.model.predict_bo(X_avail_scal)                         # (num_acq_samples, 1)
                            acq_vals = self.acq_func(mu_avail.flatten(), sigma_avail.flatten(), incumbent_scal) # (num_acq_samples,)
                        elif self.task_type == enums.TaskType.binary:
                            pred_avail, prob_avail = self.model.predict_bo(X_avail_scal)                         # (num_acq_samples, 1)
                            acq_vals = self.acq_func(prob_avail.flatten(), pred_avail.flatten(), incumbent_scal) # (num_acq_samples,)

                        if self.goal == 'minimize':
                            # higher acq_vals the better
                            sort_idxs = np.argsort(acq_vals)[::-1] # descending order
                            sample_idxs = sort_idxs[:self.batch_size]

                        elif self.goal == 'maximize':
                            # lower acq_vals the better
                            sort_idxs = np.argsort(acq_vals) # ascending order
                            sample_idxs = sort_idxs[:self.batch_size]

                        # perform measurements
                        for sample_idx in sample_idxs:
                            sample, measurement = self.sample_meas_acq(avail_df, sample_idx)
                            observations.append({'smi': sample, 'y': measurement})


                        # make a prediction on the heldout test set
                        if eval_metrics:
                            test_set_metrics = predict_test_set(
                                self.model, self.test_X, self.test_y, X_scaler, y_scaler, self.config,
                            )
                            test_set_metrics['iter_num'] = iter_num
                            all_test_set_metrics.append(test_set_metrics)

                    iter_num += 1

                # gather metrics if needed
                if eval_metrics and self.model_type not in ['random', 'nn']:
                    all_test_set_metrics = pd.concat(all_test_set_metrics)
                    all_test_set_metrics['run'] = num_restart
                    df_metrics.append(all_test_set_metrics)
                    
                data_dict = {}
                for key in observations[0].keys():
                    data_dict[key] = [o[key] for o in observations]
                data = pd.DataFrame(data_dict)
                if self.task_type == enums.TaskType.regression:
                    if self.goal == 'minimize':
                        data['trace'] = data['y'].cummin()
                    else:
                        data['trace'] = data['y'].cummax()
                elif self.task_type == enums.TaskType.binary:
                    if self.goal == 'minimize':
                        # find the negatives (ie. y==0)
                        data['trace'] = (data['y'] == 0).astype(int).cumsum()
                    else:
                        # find hte positives (ie. y==1)
                        data['trace'] = data['y'].cumsum()

                # statistics for the run
                data['model'] = self.model_type
                data['feature'] = self.feature_type
                data['dataset'] = self.dataset_name
                data['goal'] = self.goal
                data['acq_func'] = self.acq_func_type
                data['run'] = num_restart
                data['eval'] = range(0, len(data))
                df_optimization.append(data)
                
                keep_running = False

                # except:
                #     print('Run failed. Try again with different seed.')
        
        # gather all results as dataframe and return
        df_optimization = pd.concat(df_optimization)
        if df_metrics:
            df_metrics = pd.concat(df_metrics)

        return df_optimization, df_metrics



    def sample_meas_acq(self, avail_df, idx):
        ''' obtain the molecules suggested by the acquisition function 
        '''
        return avail_df.iloc[idx, [0, 2]]

    def sample_meas_randomly(self, avail_df):
        ''' take a single random sample from the available candiates
        '''
        idx = np.random.randint(avail_df.shape[0])
        return avail_df.iloc[idx, [0, 2]]

    def split_avail(self, df, observations):
        ''' return available and measured datasets 
        '''
        obs_smi = [o['smi'] for o in observations]

        avail_df = df[~(df['smi'].isin(obs_smi))]
        meas_df  = df[df['smi'].isin(obs_smi)]

        return meas_df, avail_df

    def make_xy(self, df, num=None):
        ''' generate featues and targets given a DataFrame
        '''
        y = df['y'].values.reshape(-1, 1)
        if self.feature_type == 'graphnet':
            # special treatment for GraphTuple features
            graphnet_list = df['x'].tolist()
            if num is not None:
                graphnet_list = graphnet_list[:np.amin([num, len(graphnet_list)])]
                y = y[:np.amin([num, len(graphnet_list)])]
            else:
                pass
            X = utils_tf.concat(graphnet_list, axis=0)
        else:
            # vector-valued features
            X = np.vstack(df['x'].values)
            if num is not None:
                X = X[:np.amin([num, X.shape[0]]), :]
                y = y[:np.amin([num, X.shape[0]]), :]

        return X, y


#===========
# DEBUGGING
#===========

if __name__ == '__main__':

    DATASET_NAME = 'delaney'
    GOAL = 'minimize'
    MODEL_TYPE = 'ngboost'
    FEATURE_TYPE = 'mordred'
    ACQUISITION_TYPE = 'ucb'
    BETA = 1.

    NUM_ACQ_SAMPLES = 200
    BUDGET = 80
    BATCH_SIZE = 5
    NUM_INIT_DESIGN = 0.05


    bo_exp = SynthBO(
        dataset_name=DATASET_NAME,
        goal=GOAL,
        model_type=MODEL_TYPE,
        feature_type=FEATURE_TYPE,
        acq_func_type=ACQUISITION_TYPE,
        beta=BETA,
        num_acq_samples=NUM_ACQ_SAMPLES,
        batch_size=BATCH_SIZE,
        budget=BUDGET,
        init_design_strategy='random',
        num_init_design=NUM_INIT_DESIGN,
    )

    bo_exp.run(
        num_restarts=2,
        res_file=f'results-{DATASET_NAME}-{MODEL_TYPE}-{FEATURE_TYPE}-{ACQUISITION_TYPE}-{BETA}.pkl',
    )
