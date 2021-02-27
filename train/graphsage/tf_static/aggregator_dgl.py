import tensorflow as tf
from tensorflow.keras import layers


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
        in_neigh_feats = in_feats

        if pool_feats is not None and (aggregator_type == 'maxpool' or aggregator_type == 'meanpool'):
            in_neigh_feats = pool_feats

        if in_feats > 0:
            if aggregator_type == 'maxpool':
                self.fc_pool = layers.Dense(in_neigh_feats, use_bias=bias)
            if aggregator_type == 'meanpool':
                self.fc_pool = layers.Dense(in_neigh_feats, use_bias=bias)

            self.fc_neigh = layers.Dense(out_feats, use_bias=bias)


    def call(self, h, h_self, adj_matrix):#0.011
        #h = nf.layers[layer_id].data['activation']
        #h_self = nf.layers[layer_id + 1].data['activation']

        h = self.feat_drop(h)#apply dropout
        h_self = self.feat_drop(h_self)

        #matrix, shuffle = (nf.block_adjacency_matrix(layer_id, None))
        #assert shuffle is None

        if self._aggre_type == 'meanpool':
            if self.fc_pool is not None:
                h = tf.nn.relu(self.fc_pool(h))#pooling over neighbours

            norm_matrix = adj_matrix / tf.sparse.reduce_sum(adj_matrix, axis=0)#sum over cols
            agg_features = tf.sparse.sparse_dense_matmul(norm_matrix, h)
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

        rst = self.fc_neigh(tf.concat(values=[h_self, agg_features], axis=1))

        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)

        return rst
