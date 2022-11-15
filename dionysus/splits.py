import collections
import functools
import itertools
from typing import Tuple, Dict, List, Any, Optional

import hdbscan
import numpy as np
import sklearn.decomposition
import sklearn.preprocessing
import umap
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from skmultilearn.model_selection import IterativeStratification

from . import chem
from .enums import TaskType


def get_default_split_fn(y: np.ndarray, task_type: TaskType):
    if task_type == TaskType.regression:
        return train_test_split
    elif task_type == TaskType.binary:
        return functools.partial(train_test_split, stratify=y)
    else:
        raise ValueError(f'Not implemented for {task_type}')


def get_default_kfold_splitter(n_splits: int, task_type: TaskType):
    if task_type == TaskType.regression:
        return KFold(n_splits=n_splits, shuffle=True)
    elif task_type == TaskType.binary:
        return StratifiedKFold(n_splits=n_splits, shuffle=True)
    else:
        raise ValueError(f'Not implemented for {task_type}')


def get_cluster_labels(x: np.ndarray, max_clusters: int = 20, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Returns dim-reduced features and cluster labels."""
    print('Training dimentionality reducer')
    reducer = umap.UMAP(
        n_components=5,
        n_neighbors=10,
        min_dist=0.0,  # controls how tightly points are packed together
        metric='jaccard',
        random_state=random_state)
    x_reduced = reducer.fit_transform(x)
    print('Training clusterer')
    n_clusters = max_clusters + 1
    count = 1
    while n_clusters > max_clusters:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15 * count)
        clusterer.fit(x_reduced)
        n_clusters = np.max(clusterer.labels_)
        count += 1
        if n_clusters > max_clusters:
            print(f'Found {n_clusters} clusters, expected < {max_clusters}, increasing expected cluster size.')

    cluster_labels = clusterer.labels_
    print(f'Found {n_clusters} clusters')
    counts = np.array(list(collections.Counter(clusterer.labels_).values()))[1:]
    print(f'Cluster counts: [{np.min(counts)}, {int(np.median(counts))}, {np.max(counts)}] min/median/max')
    print(f'{np.sum(cluster_labels == -1)} unassigned molecules')
    return x_reduced, cluster_labels


def build_joint_labels(fps: np.ndarray, y: np.ndarray, task_type: TaskType,
                       fp_label_dim: Optional[int] = None,
                       fp_label_threshold: float = .1) -> np.ndarray:
    if task_type == TaskType.regression:
        preproc = sklearn.preprocessing.KBinsDiscretizer(n_bins=10, encode='onehot-dense')
        labels = preproc.fit_transform(y)
    elif task_type == TaskType.binary:
        labels = y
    else:
        raise ValueError(f'Not implemented for {task_type}')
    fp_label_dim = fp_label_dim or (25 if len(fps) >= 100 else 10)
    lda = sklearn.decomposition.LatentDirichletAllocation(n_components=fp_label_dim)
    topics = lda.fit_transform(fps)
    struct_labels = topics >= fp_label_threshold
    return np.column_stack([struct_labels, labels])


def get_cluster_splits(cluster_labels: np.ndarray, n_samples: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """"Get train/test cluster splits."""
    n_clusters = np.max(cluster_labels)
    if n_clusters > 25:
        raise ValueError('Number of clusters is high, consider less clusters.')
    cluster_mask = np.array([cluster_labels == i for i in range(n_clusters)])
    no_label = cluster_labels == -1
    indices = np.arange(len(cluster_labels))
    c_indices = np.arange(n_clusters)
    splits = {}

    # First we pick all single and every but splits.
    for i in range(n_clusters):
        mask = np.logical_or.reduce([no_label, cluster_mask[i]])
        train = indices[mask]
        test = indices[np.logical_not(mask)]
        splits[f'{i}'] = (train, test)
        splits[f'all but {i}'] = (test, train)

    # We subsample combinations of clusters since the number explodes really fast.
    for n_pieces in range(2, n_clusters - 1):
        combos = list(itertools.combinations(c_indices, n_pieces))
        pick_indices = np.random.choice(np.arange(len(combos)), n_samples)
        picks = [combos[i] for i in pick_indices]
        for sample in picks:
            mask = np.logical_or.reduce([no_label] + [cluster_mask[i] for i in sample])
            train = indices[mask]
            test = indices[np.logical_not(mask)]
            name = ' & '.join(np.array(sample).astype(str)).strip()
            splits[name] = (train, test)
    return splits


def get_it_split(y, train_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Iterative stratification split of the second order.."""
    indices = np.arange(len(y))
    stratifier = IterativeStratification(n_splits=2, order=2,
                                         sample_distribution_per_fold=[1.0 - train_size, train_size])
    train_indices, test_indices = next(stratifier.split(indices, y))
    return train_indices, test_indices


def get_tvt_cluster_splits(smi: np.ndarray, y: np.ndarray,
                           cluster_labels: np.ndarray,
                           task_type: TaskType,
                           val_size: float = .15,
                           n_samples: int = 3) -> List[Dict[str, Any]]:
    """Gets Train-Val-Test cluster splits."""
    fps = chem.manysmi_to_fps(smi)
    joint_labels = build_joint_labels(fps, y, task_type)

     # add cluster label for diverse split 
    joint_cluster_labels = np.column_stack([joint_labels, cluster_labels]) 
    trainval, test = get_it_split(joint_cluster_labels, train_size=0.8)

    # cluster splits on remaining results
    tt_splits = get_cluster_splits(cluster_labels[trainval], n_samples)

    splits = []
    for name, (trainval, _) in tt_splits.items():
        train, val = get_it_split(joint_labels[trainval], train_size=1.0 - val_size)
        splits.append({'name': name, 'train': smi[train], 'val': smi[val], 'test': smi[test]})
    return splits
