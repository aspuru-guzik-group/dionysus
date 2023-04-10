# DIONYSUS: Calibration and generalizability of probabilistic models on low-data chemical datasets

**Repository is under construction, and will be continually improved**

This package is accompanied by this paper: [https://arxiv.org/abs/2212.01574](https://arxiv.org/abs/2212.01574).

Authors: Gary Tom, Riley J. Hickman, Aniket Zinzuwadia, Afshan Mohajeri, Benjamin Sanchez-Lengeling, Alan Aspuru-Guzik. (2022)

## Preliminary

DIONYSUS requires the following:
- Python >= 3.8
- rdkit >= 2021.03.3
- tensorflow == 2.9.0
- tensorflow-probability == 0.17.0
- tf-models-official == 2.7.1
- graph_nets == 1.1.0
- sonnet == 2.0.0
- ngboost == 0.3.12
- gpflow == 2.5.2
- scikit-learn == 1.0.2
- scikit-multilearn == 0.2.0
- mordred == 1.2.0
- dask == 2022.6.1
- umap-learn == 0.5.1
- cairosvg == 2.5.2
- pandas == 1.4.1
- seaborn == 0.11.2
- hdbscan == 0.8.27


## Datasets

Datasets used are shown in `data/` directory. For new datasets, simply add it to `data/raw/`, and the corresponding information in `data/dataset_registry_raw.csv`. The following information is required:

- `name`: name of dataset
- `shortname`: name that will be referenced in the scripts
- `filename`: the csv file name (`{shortname}.csv`)
- `task`: either regression/binary (classification)
- `smiles_column`: name of column containing the smiles representations
- `target_columns`: name of column containing the target of interest


## Usage

### Experiment 1: Supervised learning

Scripts to run experiments are contained in `scripts/` directory. The structure will be changed in future versions.

```bash
cd scripts/

# prepare molecules and analysis directory in data/{shortname}
python make_qsar_ready.py --dataset={shortname} 

# create splits and features
python make_features.py --dataset={shortname} 

# make all predictions for specified feature/model
python make_predictions.py --dataset={shortname} --model={modelname} --feature={featurename} 

# create evaluations for all available prediction data
python make_evaluations.py --dataset={shortname}
```

### Experiment 2: Bayesian optimization

Scripts are found in `bo/` directory. 

**More info coming**

### Experiment 3: Cluster splits and generalizability

Similar to the first experiment, the scripts are found in `scripts/` directory. Once datasets are cleaned and features are made, you can make the cluster splits, and run for specified feature/model. In the manuscript, we only do this for mordred/GPs.

All results will be contained in `data/generalization/{shortname}`.

```bash
cd scripts/

# create the cluster splits
python make_generalization_splits.py --dataset={shortname} 

# create evaluations for all available prediction data
python make_generalization_predictions.py --dataset={shortname}

# evaluate the predictions on each cluster split, and generate plots
python make_generalization_evaluations.py --dataset={shortname}
```



## Proposed structure of repository (TODO)

Structure of repository
- `mol_data`: dataset preprocessor and loader. And be extended to other datasets and new features.
- `dionysus`: model-agnostic evaluation script and library. Just requires predictions.
- `dionysus_addons`: models used in the paper, here for reproducibility.


...



