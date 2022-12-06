"""Train and make all predictions."""
import os
import sys
import warnings
import timeit

warnings.simplefilter("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT_DIR = '..'
sys.path.append(ROOT_DIR)

import numpy as np
from absl import app
from absl import flags

import dionysus
from dionysus.models import SNGP, NGB, GP, GNNGP, BNN

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', None, 'Dataset to process')
flags.DEFINE_bool('overwrite', False, 'Re-calculate and overwrite if files already exist')
flags.DEFINE_string('split_type', 'cluster', 'Types: cluster, incremental-diverse, incrememntal-standard. Default cluster.')
flags.DEFINE_string('cluster_id', None, 'Single cluster id. Defaults to loop through all clusters.')
flags.DEFINE_bool('verbose', True, 'Print out training progress.')
flags.DEFINE_integer('seed', 0, 'Random seed. Default 0.')
flags.DEFINE_integer('task_index', -1, 'Select task for multi-target dataset. All tasks set to -1.')


def check_exists(fname: str, overwrite: bool):
    if os.path.exists(fname) and not overwrite:
        print(f'{fname} already exists. Skipping...')
        return True
    else:
        return False


def main(_):
    # set random seed
    dionysus.utils.set_random_seed(FLAGS.seed)

    base_dir = os.path.join(dionysus.files.get_data_dir(ROOT_DIR), FLAGS.dataset)
    _, config, _ = dionysus.datasets.load_dataset(FLAGS.dataset, base_dir)
    config.task = config.tasks[FLAGS.task_index]
    work_dir = os.path.join(base_dir, 'generalization')

    def get_filename(ftype, fname_args=None):
        return dionysus.files.get_filename(work_dir, ftype, fname_args)

    assert FLAGS.split_type in ['cluster', 'incremental-diverse', 'incremental-standard'], 'Invalid cluster type.'

    if FLAGS.cluster_id is None:
        fname = get_filename('split', [f'{FLAGS.split_type}*', 'tvt'])
    else:
        fname = get_filename('split', [f'{FLAGS.split_type}{FLAGS.cluster_id}', 'tvt'])
    split_fnames = dionysus.utils.get_matching_files(fname)
    split_type_ls = []
    for f in split_fnames:
        split_type_ls.append(f.split('/')[-1].split('_')[0])
    print(f'Found {len(split_fnames)} splits')

    # Loop through all split types
    for j, split_type in enumerate(split_type_ls):
        start = timeit.default_timer()
        # Loop through the vector features and use corresponding models
        for feat in dionysus.enums.VECTOR_FEATURES:

            ##################################
            # BNN
            print('**** Training BNN ****')
            smi_split, x, y = dionysus.datasets.load_task(FLAGS.dataset, feat, dionysus.enums.Models.bnn,
                                                        base_dir, task_index=FLAGS.task_index, result_dir=work_dir,
                                                        split_type=split_type)
            pred_file = get_filename('predictions', [f'bnn-{split_type}', feat])

            if not check_exists(pred_file, FLAGS.overwrite):
                hp = dionysus.models.bnn.default_hp(config.task, output_dim=y.values.shape[-1])
                features_scaler, targets_scaler = dionysus.datasets.scaling_options(feat, dionysus.enums.Models.bnn, config.task)
                hp.scale_features = features_scaler is not None
                hp.scale_targets = targets_scaler is not None

                # create model, train and predict
                model = BNN.from_hparams(hp)
                model.train(x, y, hp, verbose=FLAGS.verbose)
                uniq_smi, y_mu, y_std = model.predict(smi_split, x, y)

                # save results and model 
                ground_truth = y.scaled_values if hp.scale_targets else y.values
                np.savez_compressed(pred_file, isosmiles=uniq_smi, y_pred=y_mu, y_true=ground_truth, y_err=y_std)
                # import pdb; pdb.set_trace()
            
            ##################################
            # SNGP
            print('**** Training SNGP ****')
            smi_split, x, y = dionysus.datasets.load_task(FLAGS.dataset, feat, dionysus.enums.Models.sngp,
                                                            base_dir, task_index=FLAGS.task_index, result_dir=work_dir,
                                                            split_type=split_type)
            pred_file = get_filename('predictions', [f'sngp-{split_type}', feat])

            if not check_exists(pred_file, FLAGS.overwrite):
                hp = dionysus.models.sngp.default_hp(config.task, output_dim=y.values.shape[-1])
                features_scaler, targets_scaler = dionysus.datasets.scaling_options(feat, dionysus.enums.Models.sngp, config.task)
                hp.scale_features = features_scaler is not None
                hp.scale_targets = targets_scaler is not None

                # create model, train and predict
                model = SNGP.from_hparams(hp)
                model.train(x, y, hp, verbose=FLAGS.verbose)
                uniq_smi, y_mu, y_std = model.predict(smi_split, x, y)

                # save results and model 
                ground_truth = y.scaled_values if hp.scale_targets else y.values
                np.savez_compressed(pred_file, isosmiles=uniq_smi, y_pred=y_mu, y_true=ground_truth, y_err=y_std)

            ################################
            # NGBoost
            print('**** Training NGBoost ****')
            smi_split, x, y = dionysus.datasets.load_task(FLAGS.dataset, feat, dionysus.enums.Models.ngboost,
                                                            base_dir, task_index=FLAGS.task_index, result_dir=work_dir,
                                                            split_type=split_type)
            pred_file = get_filename('predictions', [f'ngboost-{split_type}', feat])

            if not check_exists(pred_file, FLAGS.overwrite):
                hp = dionysus.models.ngb.default_hp(config.task, output_dim=y.values.shape[-1])
                model = NGB.from_hparams(hp)
                model.train(x, y, hp)
                uniq_smi, y_mu, y_std = model.predict(smi_split, x, y)

                np.savez_compressed(pred_file, isosmiles=uniq_smi, y_pred=y_mu, y_true=y.values, y_err=y_std)

            #################################
            # GP with standard kernels and likelihoods
            print('**** Training GP ****')
            smi_split, x, y = dionysus.datasets.load_task(FLAGS.dataset, feat, dionysus.enums.Models.gp,
                                                            base_dir, task_index=FLAGS.task_index, result_dir=work_dir,
                                                            split_type=split_type)
            pred_file = get_filename('predictions', [f'gp-{split_type}', feat])

            if not check_exists(pred_file, FLAGS.overwrite):
                hp = dionysus.models.gp.default_hp(config.task, input_dim=x.values.shape[-1], output_dim=y.values.shape[-1])
                hp.kernel = 'tanimoto' if feat == dionysus.enums.FeatureType.mfp else 'rbf'

                # set scalings
                features_scaler, targets_scaler = dionysus.datasets.scaling_options(feat, dionysus.enums.Models.gp, config.task)
                hp.scale_features = features_scaler is not None
                hp.scale_targets = targets_scaler is not None

                model = GP.from_hparams(hp)
                model.train(x, y, hp, verbose=FLAGS.verbose)
                uniq_smi, y_mu, y_std = model.predict(smi_split, x, y)

                ground_truth = y.scaled_values if hp.scale_targets else y.values
                np.savez_compressed(pred_file, isosmiles=uniq_smi, y_pred=y_mu, y_true=ground_truth, y_err=y_std)

        # loop through the graph features
        for feat in dionysus.enums.GRAPH_FEATURES:

            ##################################
            # GNNGP model
            print('**** Training GNNGP ****')
            smi_split, x, y = dionysus.datasets.load_task(FLAGS.dataset, feat, dionysus.enums.Models.gnngp,
                                                            base_dir, task_index=FLAGS.task_index, result_dir=work_dir,
                                                            split_type=split_type)
            pred_file = get_filename('predictions', [f'gnngp-{split_type}', feat])

            if not check_exists(pred_file, FLAGS.overwrite):
                hp = dionysus.models.gnngp.default_hp(config.task, output_dim=y.values.shape[-1])

                # set scalings
                features_scaler, targets_scaler = dionysus.datasets.scaling_options(feat, dionysus.enums.Models.gnngp,
                                                                                config.task)
                hp.scale_features = features_scaler is not None
                hp.scale_targets = targets_scaler is not None

                model = GNNGP.from_hparams(hp)
                model.train(x, y, hp, verbose=FLAGS.verbose)
                uniq_smi, y_mu, y_std = model.predict(smi_split, x, y)

                ground_truth = y.scaled_values if hp.scale_targets else y.values
                np.savez_compressed(pred_file, isosmiles=uniq_smi, y_pred=y_mu, y_true=ground_truth, y_err=y_std)
    
        stop = timeit.default_timer()
        print(f'{j+1}/{len(split_type_ls)} completed. Time: ', stop - start)  
        # import pdb;  pdb.set_trace()



if __name__ == '__main__':
    flags.mark_flag_as_required('dataset')
    app.run(main)
