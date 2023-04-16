import os, sys
import warnings

warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT_DIR = '..'
sys.path.append(ROOT_DIR)

from absl import app, flags
import dionysus
from dionysus.bo import SynthBO
import seaborn as sns
import matplotlib.pyplot as plt


FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', None, 'Dataset to process')
flags.DEFINE_string('model', 'ngboost', 'Model to run BO experiment.')
flags.DEFINE_string('feature', 'mfp', 'Feature to use for BO experiment.')
flags.DEFINE_string('goal', 'minimize', 'Whether to maximize or minimize.')
flags.DEFINE_string('acquisition', 'ucb', 'Acquisition function to use.')
flags.DEFINE_float('beta', 0.2, 'Beta parameter for UCB. Ignored if not UCB.')
flags.DEFINE_bool('overwrite', False, 'Re-calculate and overwrite if files already exist')
flags.DEFINE_bool('verbose', True, 'Print out training progress.')
flags.DEFINE_integer('task_index', -1, 'Select task for multi-target dataset. All tasks set to -1.')

# settings for BO runs
flags.DEFINE_integer('num_restarts', 30, 'Number of repeated runs (different seeds).')
flags.DEFINE_integer('num_acq_samples', 1000, 'Number of points to evaluate acquisition function.')
flags.DEFINE_integer('budget', 250, 'Budget/max number of iterations of BO.')
flags.DEFINE_integer('batch_size', 5, 'Number of molecules sampled each iteration.')
flags.DEFINE_float('frac_init_design', 0.05, 'Percentage of dataset to start with. Minimum 25, maximum 100 molecules.')

def check_exists(fname: str, overwrite: bool):
    if os.path.exists(fname) and not overwrite:
        print(f'{fname} already exists. Skipping...')
        return True
    else:
        return False

def main(_):
    # working directory for multiple runs 
    work_dir = os.path.join(dionysus.files.get_data_dir(ROOT_DIR), FLAGS.dataset)

    # output directory for bo experiments
    out_dir = os.path.join(work_dir, 'bayesian_optimization')
    os.makedirs(out_dir, exist_ok=True)

    trace_file = dionysus.files.get_filename(out_dir, 'bo_traces', [FLAGS.model, FLAGS.feature])
    metric_file = name = dionysus.files.get_filename(out_dir, 'bo_metrics', [FLAGS.model, FLAGS.feature])

    bo_exp = SynthBO(
        dataset_name=FLAGS.dataset,
        goal=FLAGS.goal,
        model_type=FLAGS.model,
        feature_type=FLAGS.feature,
        acq_func_type=FLAGS.acquisition,
        beta=FLAGS.beta,
        num_acq_samples=FLAGS.num_acq_samples,
        batch_size=FLAGS.batch_size,
        budget=FLAGS.budget,
        init_design_strategy='random',
        num_init_design=FLAGS.frac_init_design,
        work_dir = work_dir
    )

    if not check_exists(trace_file, FLAGS.overwrite):
        df_opt, df_metric = bo_exp.run(
            num_restarts=FLAGS.num_restarts, eval_metrics=False
        )
        df_opt.to_csv(trace_file, index=False)
        if df_metric:
            df_metric.to_csv(metric_file, index=False)




if __name__ == '__main__':
    flags.mark_flag_as_required('dataset')
    app.run(main)
