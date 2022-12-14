#!/usr/bin/env python

import os, sys
ROOT_DIR = '../../../../'
BO_DIR  = '../../../'
sys.path.append(ROOT_DIR)
sys.path.append(BO_DIR)

from synthbo import SynthBO

DATASET_NAME = sys.argv[1]
GOAL = sys.argv[2]
MODEL_TYPE = sys.argv[3]
FEATURE_TYPE = sys.argv[4]
ACQUISITION_TYPE = sys.argv[5]
BETA = float(sys.argv[6])
NUM_ACQ_SAMPLES = 500
BUDGET = 150
BATCH_SIZE = 5
NUM_INIT_DESIGN = 20


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
    num_restarts=20,
    res_file='results.pkl',
)
