import graph_nets
import ml_collections
import sonnet as snt
import tensorflow as tf

from . import modules
from .. import enums, types


class NodesAggregator(snt.Module):
    """Aggregates neighboring nodes based on sent and received nodes."""

    def __init__(self,
                 reducer=tf.math.unsorted_segment_sum,
                 name='nodes_aggregator'):
        super(NodesAggregator, self).__init__(name=name)
        self.reducer = reducer

    def __call__(self, graph: types.GraphsTuple) -> tf.Tensor:
        num_nodes = tf.reduce_sum(graph.n_node)
        adjacent_nodes = tf.gather(graph.nodes, graph.senders)
        return self.reducer(adjacent_nodes, graph.receivers, num_nodes)


class NodeLayer(graph_nets.blocks.NodeBlock):
    """GNN layer that only updates nodes, but uses edges."""

    def __init__(self, *args, **kwargs):
        super(NodeLayer, self).__init__(*args, use_globals=False, **kwargs)


class GCNLayer(graph_nets.blocks.NodeBlock):
    """GNN layer that only updates nodes using neighboring nodes and edges."""

    def __init__(self, *args, **kwargs):
        super(GCNLayer, self).__init__(*args, use_globals=False, **kwargs)
        self.gather_nodes = NodesAggregator()

    def __call__(self, graph: types.GraphsTuple) -> types.GraphsTuple:
        """Collect nodes, adjacent nodes, edges and update to get new nodes."""
        nodes_to_collect = []
        if self._use_sent_edges:
            nodes_to_collect.append(self._sent_edges_aggregator(graph))
        if self._use_received_edges:
            nodes_to_collect.append(self._received_edges_aggregator(graph))
        if self._use_nodes:
            nodes_to_collect.append(graph.nodes)

        nodes_to_collect.append(self.gather_nodes(graph))

        if self._use_globals:
            nodes_to_collect.append(graph_nets.blocks.broadcast_globals_to_nodes(graph))

        collected_nodes = tf.concat(nodes_to_collect, axis=-1)
        updated_nodes = self._node_model(collected_nodes)
        return graph.replace(nodes=updated_nodes)


class NodeEdgeLayer(snt.Module):
    """GNN layer that only updates nodes and edges."""

    def __init__(self, node_model_fn: types.ModuleMaker, edge_model_fn: types.ModuleMaker, name='NodeEdgeLayer'):
        super(NodeEdgeLayer, self).__init__(name=name)
        self.edge_block = graph_nets.blocks.EdgeBlock(
            edge_model_fn=edge_model_fn, use_globals=False)
        self.node_block = graph_nets.blocks.NodeBlock(
            node_model_fn=node_model_fn, use_globals=False)

    def __call__(self, graph: types.GraphsTuple) -> types.GraphsTuple:
        return self.node_block(self.edge_block(graph))


def get_graph_block(block_type: enums.GNNBlock, node_size: int,
                    edge_size: int, global_size: int, index: int) -> types.Module:
    """Gets a GNN block based on enum and sizes."""
    name = f'{block_type.name}_{index + 1}'
    if block_type == enums.GNNBlock.gcn:
        return GCNLayer(modules.get_mlp_fn([node_size] * 2), name=name)
    elif block_type == enums.GNNBlock.mpnn:
        return NodeEdgeLayer(
            modules.get_mlp_fn([node_size] * 2),
            modules.get_mlp_fn([edge_size] * 2),
            name=name)
    elif block_type == enums.GNNBlock.graphnet:
        use_globals = index != 0
        return graph_nets.modules.GraphNetwork(
            node_model_fn=modules.get_mlp_fn([node_size] * 2),
            edge_model_fn=modules.get_mlp_fn([edge_size] * 2),
            global_model_fn=modules.get_mlp_fn([global_size] * 2),
            edge_block_opt={'use_globals': use_globals},
            node_block_opt={'use_globals': use_globals},
            global_block_opt={'use_globals': use_globals},
            name=name)
    else:
        raise ValueError(f'block_type={block_type} not implemented')


class GNN(snt.Module):
    """A general graph neural network for graph property prediction."""

    def __init__(self,
                 node_size: int,
                 edge_size: int,
                 global_size: int,
                 output_dim: int,
                 block_type: enums.GNNBlock,
                 output_act: types.CastableActivation,
                 n_layers: int = 3):
        super(GNN, self).__init__(name=block_type.name)

        # Graph encoding step, basic linear mapping.
        self.encode = graph_nets.modules.GraphIndependent(
            node_model_fn=lambda: snt.Linear(node_size),
            edge_model_fn=lambda: snt.Linear(edge_size))
        # Message passing steps or GNN blocks.
        gnn_layers = [
            get_graph_block(
                block_type,
                node_size,
                edge_size,
                global_size,
                index)
            for index in range(0, n_layers)
        ]
        self.gnn = snt.Sequential(gnn_layers)
        self.pred_layer = modules.get_pred_layer(output_dim, output_act)

    def embed(self, x: types.GraphsTuple) -> tf.Tensor:
        return self.gnn(self.encode(x)).globals

    def __call__(self, x: types.GraphsTuple) -> tf.Tensor:
        return self.pred_layer(self.embed(x))

    @classmethod
    def from_hparams(cls, hp: types.ConfigDict) -> 'GNN':
        return cls(node_size=hp.node_size,
                   edge_size=hp.edge_size,
                   global_size=hp.global_size,
                   output_dim=hp.output_dim,
                   block_type=enums.GNNBlock(hp.block_type),
                   output_act=hp.output_act,
                   n_layers=hp.n_layers)


def default_hp(task: enums.TaskType, output_dim: int) -> types.ConfigDict:
    hp = ml_collections.ConfigDict()
    hp.node_size = 50
    hp.edge_size = 20
    hp.global_size = 150
    hp.block_type = 'graphnet'
    hp.n_layers = 3
    hp.task = str(enums.TaskType(task))
    hp.output_act = modules.task_to_activation_str(hp.task)
    hp.output_dim = output_dim
    hp.lr = 1e-3
    hp.epochs = 2000
    hp.batch_size = 256
    hp.patience = 200
    return hp
