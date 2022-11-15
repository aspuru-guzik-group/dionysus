import base64
import io
import os
from typing import Callable, Sequence

import IPython.display
import cairosvg
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit.Chem import Draw
import rdkit.Chem.Draw as rkDraw
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from . import enums, types

CMAP = {}
all_features = enums.VECTOR_FEATURES + enums.GRAPH_FEATURES
for i, f in enumerate(all_features):
    CMAP.update({f: sns.color_palette()[i]})



def get_cluster_cmap(n_clusters: int):
    return sns.color_palette('tab20', n_clusters)


def save_figure(adir, name, ext=None):
    exts = ['.png', '.svg']
    for ext in exts:
        fname = os.path.join(adir, name + ext)
        plt.savefig(fname, dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.0,
                    transparent=True)


def save_svg(svg_str, adir, name, also_png=True):
    svg_bytes = svg_str.encode()
    svg_fname = os.path.join(adir, f'{name}.svg')
    with open(svg_fname, 'w') as afile:
        afile.write(svg_str)
    if also_png:
        png_fname = os.path.join(adir, f'{name}.png')
        cairosvg.svg2png(bytestring=svg_bytes, write_to=png_fname)


def plotting_settings():
    """Change matplotlib defaults"""

    # Get some seaborn defaults.
    sns.set_style('ticks')
    sns.set_context('talk', font_scale=1.25)

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'DejaVu Sans'

    mpl.rcParams['figure.figsize'] = [12.0, 8.0]

    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['savefig.dpi'] = 72
    mpl.rcParams['savefig.pad_inches'] = 0.1
    mpl.rcParams['savefig.transparent'] = True

    mpl.rcParams['axes.linewidth'] = 2.5
    mpl.rcParams['legend.markerscale'] = 2.0
    mpl.rcParams['legend.fontsize'] = 'small'

    # Make colab/jupyter plots HD.
    IPython.display.set_matplotlib_formats('retina')


def encode_image_as_base64(img):
    """Encode image to base64 strings, useful to embed images in HTML/js."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded_string = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return encoded_string


def encode_mols_as_base64(mols, img_size: int = 128):
    """Encode molecule images as base64 strings, useful to embed images."""
    img_list = [Draw.MolToImage(m, size=(img_size, img_size)) for m in mols]
    return [encode_image_as_base64(img) for img in img_list]


def plot_calibration_diagram(df: pd.DataFrame, compare_by: str, label: str, task: enums.TaskType):
    ''' Calibration diagrams based on uncertainties.
    '''
    if task == enums.TaskType.regression:
        fig, ax = plt.subplots()
        for cat in df[compare_by].unique():
            subset = df[df[compare_by] == cat].groupby('q')['C(q)']
            means = subset.mean()
            ci_bot = subset.quantile(0.025)
            ci_top = subset.quantile(0.975)
            ax.plot(means.index, means.values, label=cat)
            ax.fill_between(means.index, ci_bot.values, ci_top.values, alpha=0.3)
        ax.plot([0, 1], [0, 1], color='k', linestyle='--')  # perfect calibration for reference
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_ylabel('C(q) score')
        ax.set_xlabel('Quantile (q)')
        ax.set_title(f'{label} calibration')
        ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1))

    elif task == enums.TaskType.binary:
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x='q', y='C(q)', hue=compare_by, style=compare_by, ci=None, markers=False, dashes=False,
                     ax=ax)
        ax.plot([0, 1], [0, 1], color='k', linestyle='--')  # perfect calibration for reference
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Confidence')
        ax.set_title(f'{label} calibration')
        ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
    else:
        raise NotImplementedError(f'{task} not implemented!')

    return fig


def plot_metrics(df: pd.DataFrame, metric: str, compare_by: str, group_by: str, task: enums.TaskType):
    if task == enums.TaskType.regression:
        fig, ax = plt.subplots()
        sns.barplot(data=df, x=group_by, y=metric, hue=compare_by, ax=ax)

        barplot_confidence_intervals(data=df, x=group_by, y=metric, hue=compare_by,
                                     err=f'{metric}_CI', ax=ax, rotate=True)
        ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
        if metric == 'R^2':
            ax.set_ylim([0, 1.0])
            ax.set_ylabel(r'$R^2$')
        elif metric == 'absCVPParea':
            ax.set_ylim([0, 0.5])
            ax.set_ylabel('Absolute Miscalibration Area')
        elif metric == 'kendall':
            ax.set_ylim([-1.0, 1.0])
            ax.set_ylabel('Kendall tau')
        

    elif task == enums.TaskType.binary or task == enums.TaskType.multiclass:
        fig, ax = plt.subplots()
        sns.barplot(data=df, x=group_by, y=metric, hue=compare_by, ax=ax)

        barplot_confidence_intervals(data=df, x=group_by, y=metric, hue=compare_by,
                                     err=f'{metric}_CI', ax=ax, rotate=True)
        ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
        if metric == 'kendall':
            ax.set_ylim([0, 1])
            ax.set_ylabel('Kendall tau')
        elif metric == 'auroc':
            ax.set_ylim([0.5, 1.0])
            ax.set_ylabel('AUROC')
        elif metric == 'ece':
            ax.set_ylabel('Expected Calibration Error')
    else:
        raise NotImplementedError(f'{task} not implemented.')

    return fig


def plot_pred_calib_comparison(df: pd.DataFrame, perf_metric: str, calib_metric: str):
    df = df.loc[:, ~df.columns.duplicated()]  # remove any duplicated columns
    df = df[~(df == 0).any(axis=1)]  # remove all rows with zero values

    fig, ax = plt.subplots()
    palette = sns.color_palette('tab10')  # to match with mpl
    g = sns.scatterplot(x=perf_metric, y=calib_metric, hue='feature', style='method', data=df, palette='tab10', ax=ax, s=200, alpha=0.8)
    ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
    if perf_metric == 'R^2':
        ax.set_xlabel(r'$R^2$')
        ax.set_xlim([0.3, 1.0])
    elif perf_metric == 'auroc':
        ax.set_xlabel('AUROC')
        ax.set_xlim([0.7, 1.0])

    if calib_metric == 'absCVPParea':
        ax.set_ylabel('Absolute Miscalibration Area')
        ax.set_ylim([0.0, 0.4])
    elif calib_metric == 'ece':
        ax.set_ylabel('Expected Calibration Error')
        ax.set_ylim([0.0, 0.4])

    for i, m in enumerate(df['feature'].unique()):
        subdf = df[df['feature'] == m]
        plt.errorbar(subdf[perf_metric], subdf[calib_metric],
                     xerr=subdf[[perf_metric + '_CI_bot', calib_metric + '_CI_top']].values.T,
                     yerr=subdf[[perf_metric + '_CI_bot', calib_metric + '_CI_top']].values.T, fmt='none', zorder=-100,
                     alpha=0.3,
                     color=palette[i])

    return fig


##### HELPER FUNCTIONS #####   

def barplot_confidence_intervals(data: pd.DataFrame, x: str, y: str, hue: str,
                                 err: str, ax: plt.Axes, rotate: bool = True):
    u = data[x].unique()
    x_loc = np.arange(len(u))
    subx = data[hue].unique()
    offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.)
    width = 0 if len(offsets) == 1 else np.diff(offsets).mean()
    for i, gr in enumerate(subx):
        dfg = data[data[hue] == gr]
        ax.bar(x_loc + offsets[i], dfg[y].values, width=width,
               yerr=[dfg[err + '_bot'].values, dfg[err + '_top'].values])
    ax.set_ylabel(y)
    ax.set_xticks(x_loc)

    if rotate:
        ax.set_xticklabels(u, rotation=90)
    else:
        ax.set_xticklabels(u)


def plot_cluster_index(cluster_labels: np.ndarray, name: str, cmap=None, size: int = 1, ax=None) -> None:
    """Draw clusters index map."""
    n_clusters = np.max(cluster_labels)
    cmap = cmap or get_cluster_cmap(n_clusters)
    n = n_clusters + 1
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    pos = np.arange(0, n)
    ax.imshow(pos.reshape(1, -1),
            #   cmap=mpl.colors.ListedColormap([(1.0, 1.0, 1.0)] + list(cmap)),
              cmap=mpl.colors.ListedColormap(list(cmap)),
              interpolation="nearest", aspect="auto")

    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.xaxis.set_major_locator(ticker.NullLocator())
    for i in pos:
        plt.text(i, -0.1, f'{i:d}', va='center', ha='center', fontsize='large')
        count = np.sum(cluster_labels == i)
        plt.text(i, .35, f'({count:d})', va='center', ha='center', fontsize='small')

    plt.title(f'Cluster colormap index (count) for {name}')


def plot_cluster_space(x_reduced: np.ndarray, cluster_labels: np.ndarray, name: str, cmap=None):
    """Plot the space of cluster points."""
    n_clusters = np.max(cluster_labels)
    has_label = cluster_labels != -1
    no_label = np.logical_not(has_label)
    cmap = mpl.colors.ListedColormap(cmap or get_cluster_cmap(n_clusters))
    plt.figure(figsize=(12, 12))
    plt.scatter(x_reduced[no_label, 0], x_reduced[no_label, 1], s=10, c='w', linewidth=0.5, edgecolors='0.2')
    plt.scatter(x_reduced[has_label, 0], x_reduced[has_label, 1], s=8, c=cluster_labels[has_label], cmap=cmap,
                alpha=0.75)
    plt.axis('equal')
    plt.title(f'UMAP for {name}, colored by cluster index')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    sns.despine()


def plot_cluster_counts(cluster_labels: np.ndarray, name: str, cmap=None):
    """Draw counts per cluster"""
    n_clusters = np.max(cluster_labels)
    cmap = cmap or get_cluster_cmap(n_clusters)
    cmap = mpl.colors.ListedColormap([(1.0, 1.0, 1.0)] + list(cmap))
    sns.countplot(x=cluster_labels, palette=cmap.colors, edgecolor='k')
    plt.xlabel('Cluster index')
    plt.title(f'Cluster count for {name}')
    sns.despine()


def draw_cluster_mols(mols: Sequence[types.Mol], cluster_labels: np.ndarray, n_mols_per_cluster: int = 3,
                      img_size: int = 400):
    """Draw cluster mols."""
    cluster_mols = []
    legends = []
    for i in np.unique(cluster_labels):
        mask = cluster_labels == i
        cluster = mols[mask]
        cluster_mols.extend(np.random.choice(cluster, n_mols_per_cluster).tolist())
        legends.extend([str(i)] * n_mols_per_cluster)

    svg = rkDraw.MolsToGridImage(cluster_mols, molsPerRow=n_mols_per_cluster, subImgSize=(img_size, img_size),
                                 legends=legends, useSVG=True)
    return svg


DEFUALT_N_GRID = 100
DEFAULT_X_RANGE = [0, 1]
DEFAULT_Y_RANGE = [0, 1]


def get_embeddings_background(x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE, n_grid: int = DEFUALT_N_GRID):
    # create mesh grid background
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    xv, yv = np.meshgrid(x, y)

    return np.stack([xv.flatten(), yv.flatten()], axis=-1)


def plot_embeddings(embeddings, predictions, uncertainties, predictor: Callable):
    pca = PCA(n_components=2)
    feature_scaler = MinMaxScaler()
    # target_scaler = MinMaxScaler()
    # std_scaler = MinMaxScaler()

    # get pca features
    features = pca.fit_transform(embeddings)
    features = feature_scaler.fit_transform(features)

    # generate test points on a grid
    test_grid = get_embeddings_background()
    test_grid = feature_scaler.inverse_transform(test_grid)
    test_grid = pca.inverse_transform(test_grid)
    y_pred, y_std = predictor(test_grid)

    y_pred = y_pred.numpy()
    y_std = y_std.numpy()

    targets = predictions
    stds = uncertainties

    # scale the targets
    # largest_target = max(np.append(y_pred, predictions))
    # largest_std = max(np.append(y_std, uncertainties))
    # print(largest_target)
    # print(largest_std)

    # y_pred /= largest_target
    # targets = predictions / largest_target
    # stds = uncertainties / largest_std
    # y_std /= largest_std

    # target_scaler.fit(np.append(y_pred, predictions, axis=0))
    # y_pred = target_scaler.transform(y_pred)
    # targets = target_scaler.transform(predictions)

    # std_scaler.fit(np.append(y_std, uncertainties, axis=0))
    # y_std = std_scaler.transform(y_std)
    # stds = std_scaler.transform(uncertainties)

    # plot predictions
    plt.scatter(features[:, 0], features[:, 1], c=targets, alpha=0.5, marker='.')
    plt.imshow(
        np.reshape(y_pred, [DEFUALT_N_GRID, DEFUALT_N_GRID]),
        origin='lower',
        extent=DEFAULT_X_RANGE + DEFAULT_Y_RANGE,
        interpolation='bicubic',
        aspect='auto'
    )
    plt.colorbar()
    plt.show()

    # plot uncertainties
    plt.scatter(features[:, 0], features[:, 1], c=stds, alpha=0.5, marker='.')
    plt.imshow(
        np.reshape(y_std, [DEFUALT_N_GRID, DEFUALT_N_GRID]),
        origin="lower",
        extent=DEFAULT_X_RANGE + DEFAULT_Y_RANGE,
        interpolation='bicubic',
        aspect='auto'
    )
    plt.colorbar()
    plt.show()

    return
