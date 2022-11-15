import dataclasses
from typing import Dict, List, Text, Union

import graph_nets
import numpy as np
import tensorflow as tf
import tree

from . import chem
from . import types

UNK = 'other'

Item = Union[int, str, float]

# Alias to mirror the tf version.
cast_to_np = graph_nets.utils_tf.nest_to_numpy


def cast_to_tf(graphs: types.GraphsTuple) -> types.GraphsTuple:
    """Convert GraphsTuple numpy arrays to tf.Tensor."""

    def cast_fn(x):
        return tf.convert_to_tensor(x) if isinstance(x, np.ndarray) else x

    return tree.map_structure(cast_fn, graphs)


def print_graphs_tuple(graphs: types.GraphsTuple):
    """Print a graph tuple's shapes and contents."""
    print("Shapes of GraphsTuple's fields:")
    print(
        graphs.map(
            lambda x: x if x is None else x.shape,
            fields=graph_nets.graphs.ALL_FIELDS))


def get_graphs(graphs: types.GraphsTuple, indices: np.ndarray) -> types.GraphsTuple:
    """Gets a new graphstuple (tf) based on a list of indices."""
    node_indices = tf.concat(
        [tf.constant([0]), tf.cumsum(graphs.n_node)], axis=0)
    node_starts = tf.gather(node_indices, indices)
    node_ends = tf.gather(node_indices, indices + 1)
    node_slice = tf.ragged.range(node_starts, node_ends).values
    nodes = tf.gather(graphs.nodes, node_slice)

    edge_indices = tf.concat(
        [tf.constant([0]), tf.cumsum(graphs.n_edge)], axis=0)
    edge_starts = tf.gather(edge_indices, indices)
    edge_ends = tf.gather(edge_indices, indices + 1)
    edge_slice = tf.ragged.range(edge_starts, edge_ends).values

    edges = tf.gather(graphs.edges,
                      edge_slice) if graphs.edges is not None else None

    n_edge = tf.gather(graphs.n_edge, indices)
    n_node = tf.gather(graphs.n_node, indices)

    offsets = tf.repeat(node_starts, tf.gather(graphs.n_edge, indices))
    senders = tf.gather(graphs.senders, edge_slice) - offsets
    receivers = tf.gather(graphs.receivers, edge_slice) - offsets
    new_offsets = tf.concat([tf.constant([0]), tf.cumsum(n_node)], axis=0)
    senders = senders + tf.repeat(new_offsets[:-1], n_edge)
    receivers = receivers + tf.repeat(new_offsets[:-1], n_edge)

    g_globals = tf.gather(graphs.globals,
                          indices) if graphs.globals is not None else None

    return graph_nets.graphs.GraphsTuple(
        nodes=nodes,
        edges=edges,
        globals=g_globals,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge)


@dataclasses.dataclass
class GraphSplit(types.DataSplit):
    values: types.GraphsTuple
    split: types.IndexSplit

    def get_data_split(self, indices: types.Indices) -> types.GraphsTuple:
        return get_graphs(self.values, indices)


def encode_onehot(x: Item, allowable_set: List[Item]) -> List[bool]:
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == item for item in allowable_set]


class MolTensorizer(object):
    """MolTensorizer: Convert data into molecular tensors."""

    def __init__(self, atom_set: List[str]):
        self.atom_set = atom_set + ['other']
        mol = chem.smi_to_mol('CC')
        self.node_ndim = self.get_node_features(mol.GetAtomWithIdx(0)).shape[-1]
        self.edge_ndim = self.get_edge_features(mol.GetBondWithIdx(0)).shape[-1]

    def get_node_features(self, atom: types.Atom) -> np.ndarray:
        values = [encode_onehot(atom.GetSymbol(), self.atom_set),
                  encode_onehot(str(atom.GetChiralTag()),
                                ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER']),
                  encode_onehot(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, UNK]),
                  encode_onehot(atom.GetFormalCharge(), [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, UNK]),
                  encode_onehot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, UNK]),
                  encode_onehot(atom.GetNumRadicalElectrons(), [0, 1, 2, 3, 4, UNK]),
                  encode_onehot(str(atom.GetHybridization()), ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', UNK]),
                  [atom.GetIsAromatic(), atom.IsInRing()]]
        return np.hstack(values)

    def get_edge_features(self, bond: types.Bond) -> np.ndarray:
        values = [encode_onehot(str(bond.GetBondType()), ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', UNK]),
                  encode_onehot(str(bond.GetStereo()), ['STEREONONE', 'STEREOZ',
                                                        'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY']),
                  [bond.GetIsConjugated()]]
        return np.hstack(values)

    def mol_to_data_dict(self, mol: types.Mol) -> Dict[Text, np.ndarray]:
        """Gets data dict from a single mol."""
        nodes = np.array([self.get_node_features(atom) for atom in mol.GetAtoms()])
        edges = np.zeros((mol.GetNumBonds() * 2, self.edge_ndim))
        senders = []
        receivers = []
        for index, bond in enumerate(mol.GetBonds()):
            id1 = bond.GetBeginAtom().GetIdx()
            id2 = bond.GetEndAtom().GetIdx()
            bond_arr = self.get_edge_features(bond)
            edges[index * 2, :] = bond_arr
            edges[index * 2 + 1, :] = bond_arr
            senders.extend([id1, id2])
            receivers.extend([id2, id1])
        data_dict = {
            'nodes': nodes.astype(np.float32),
            'edges': edges.astype(np.float32),
            'globals': np.array([0.], dtype=np.float32),
            'senders': np.array(senders, np.int32),
            'receivers': np.array(receivers, np.int32)
        }
        return data_dict

    def __call__(self, smiles_list: types.TextArray) -> types.GraphsTuple:
        """Transform to data dicts, useful with graph_nets library."""
        mol_list = [chem.smi_to_mol(item) for item in smiles_list]
        data_dicts = [self.mol_to_data_dict(m) for m in mol_list]
        return graph_nets.utils_tf.data_dicts_to_graphs_tuple(data_dicts)
