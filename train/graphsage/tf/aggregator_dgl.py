import tensorflow as tf
from tensorflow.keras import layers

def concatenate_edge_feats(edges):
    if "feat" not in edges.data:
        return {'m': edges.src['h']}
    x = edges.data["feat"]

    if isinstance(x, tf.SparseTensor):
        x = tf.sparse.to_dense(x)
    return {'m' : tf.concat([edges.src['h'], edges.data["feat"]], 1)}

class SAGEConv(layers.Layer):
    r"""GraphSAGE layer from paper `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__.
    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} & = \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)
        h_{i}^{(l+1)} & = \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1} + b) \right)
        h_{i}^{(l+1)} & = \mathrm{norm}(h_{i}^{l})
    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size.
        If the layer is to be applied on a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
        If aggregator type is ``gcn``, the feature size of source and destination nodes
        are required to be the same.
    out_feats : int
        Output feature size.
    feat_drop : float
        Dropout rate on features, default: ``0``.
    aggregator_type : str
        Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 edge_feats=None,
                 activation=None,
                 pool_feats = None):
        super(SAGEConv, self).__init__()

        self._in_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = layers.Dropout(feat_drop)
        self.activation = activation

        self.fc_pool = None
        self.lstm = None
        self.fc_self = None
        self.fc_neigh = None

        if edge_feats is None:
            edge_feats = 0

        if aggregator_type == 'lstm' and in_feats + edge_feats > 0:
            self.lstm = layers.LSTM(in_feats + edge_feats)

        in_neigh_feats = in_feats

        if pool_feats is not None and (aggregator_type == 'maxpool' or aggregator_type == 'meanpool'):
            in_neigh_feats = pool_feats

        if in_feats > 0:
            if aggregator_type == 'maxpool':
                self.fc_pool = layers.Dense(in_neigh_feats, use_bias=bias)
            if aggregator_type == 'meanpool':
                self.fc_pool = layers.Dense(in_neigh_feats, use_bias=bias)

            self.fc_neigh = layers.Dense(out_feats, use_bias=bias)
        self.in_neigh_feats = in_neigh_feats

    def _lstm_reducer(self, nodes):
        m = nodes.mailbox['m']  # (B, L, D)
        rst = self.lstm(m)
        return {'neigh': rst}

    def call(self, graph, feat):#0.011

        with graph.local_scope():
            feat_src = feat_dst = self.feat_drop(feat)
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]

            h_self = feat_dst
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = tf.zeros(
                    feat_dst.shape[0], self.in_neigh_feats).to(feat_dst)

            if self._aggre_type == 'mean':
                graph.srcdata['h'] = feat_src
                graph.update_all(concatenate_edge_feats, lambda node: {'neigh': tf.math.reduce_mean(node.mailbox['m'], axis=1)})
                h_neigh = graph.dstdata['neigh']

            elif self._aggre_type == 'gcn':

                graph.srcdata['h'] = feat_src
                graph.dstdata['h'] = feat_dst
                graph.update_all(concatenate_edge_feats, lambda node: {'neigh': tf.math.reduce_sum(node.mailbox['m'], axis=1)})
                degs = graph.in_degrees().to(feat_dst.device)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)

            elif self._aggre_type == 'maxpool':
                if self.fc_pool is not None:
                    graph.srcdata['h'] = tf.nn.relu(self.fc_pool(feat_src))
                else:
                    graph.srcdata['h'] = feat_src

                graph.update_all(concatenate_edge_feats, lambda node: {'neigh': tf.math.reduce_max(node.mailbox['m'], axis=1)})
                h_neigh = graph.dstdata['neigh']

            elif self._aggre_type == 'meanpool':
                if self.fc_pool is not None:
                    graph.srcdata['h'] = tf.nn.relu(self.fc_pool(feat_src))
                else:
                    graph.srcdata['h'] = feat_src

                graph.update_all(concatenate_edge_feats, lambda node: {'neigh': tf.math.reduce_mean(node.mailbox['m'], axis=1)})
                h_neigh = graph.dstdata['neigh']

            elif self._aggre_type == 'lstm':
                graph.srcdata['h'] = feat_src
                graph.update_all(concatenate_edge_feats, self._lstm_reducer)
                h_neigh = graph.dstdata['neigh']

            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            if self._aggre_type == 'gcn':
                rst = self.fc_neigh(h_neigh)
            else:
                rst = self.fc_neigh(tf.concat([h_self, h_neigh], 1))


            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)

            return rst
