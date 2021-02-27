"""Torch Module for GraphSAGE layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
from torch import nn
from torch.nn import functional as F
import torch

def concatenate_edge_feats(edges):
    if "feat" not in edges.data:
        return {'m': edges.src['h']}
    x = edges.data["feat"]
    if x.is_sparse:
        x = x.to_dense().float()
    return {'m' : torch.cat((edges.src['h'], x), 1)}


class SAGEConv(nn.Module):
    """GraphSAGE layer from paper `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__.

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} & = \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

        h_{i}^{(l+1)} & = \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1} + b) \right)

        h_{i}^{(l+1)} & = \mathrm{norm}(h_{i}^{l})

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    feat_drop : float
        Dropout rate on features, default: ``0``.
    aggregator_type : str
        Aggregator type to use (``mean``, ``gcn``, ``max/mean pool``, ``lstm``).
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    edge_feats: size of the edge features
    pool_feats: size of the pooling features (only pooling aggregators)
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 edge_feats = None,
                 activation=None,
                 pool_feats = None):
        super(SAGEConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn

        self.fc_pool = None
        self.lstm = None
        self.fc_self = None
        self.fc_neigh = None

        if edge_feats is None:
            edge_feats = 0

        if aggregator_type == 'lstm' and in_feats+edge_feats > 0:
            self.lstm = nn.LSTM(in_feats+edge_feats, in_feats+edge_feats, batch_first=True)

        in_neigh_feats = in_feats
        if pool_feats is not None and (aggregator_type == 'maxpool' or aggregator_type == 'meanpool'):
            in_neigh_feats = pool_feats

        if in_feats > 0:
            if aggregator_type == 'maxpool':
                self.fc_pool = nn.Linear(in_feats, in_neigh_feats)
            if aggregator_type == 'meanpool':
                self.fc_pool = nn.Linear(in_feats, in_neigh_feats)
            #if aggregator_type != 'gcn':
            #    self.fc_self = nn.Linear(in_feats, out_feats, bias=bias)

            if aggregator_type == 'gcn':
                self.fc_neigh = nn.Linear(in_neigh_feats, out_feats, bias=bias)
            else:
                self.fc_neigh = nn.Linear(in_neigh_feats+edge_feats+in_feats, out_feats, bias=bias)
        self.in_neigh_feats = in_neigh_feats

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')

        if self.fc_pool is not None:
            if self._aggre_type == 'maxpool':
                nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
            if self._aggre_type == 'meanpool':
                nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm' and self.lstm is not None:
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn' and self.fc_self is not None:
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)

        if self.fc_neigh is not None:
            nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_feats)),
             m.new_zeros((1, batch_size, self._in_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}

    def forward(self, graph, feat):#0.004
        r"""Compute GraphSAGE layer.

        Parameters
        ----------
        graph: layered (bipartite) graph. Edges connect two sampling depths
        feat: vertex feature tensor (#vertices in graph, #features)

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """

        with graph.local_scope():
            feat_src = feat_dst = self.feat_drop(feat)#apply dropout (train phase). Feat_src = features of the 'leaf' vertice
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()] #feat_dst: features of the 'parent' vertices.

            h_self = feat_dst

            # Handle the case of graphs without edges (e.g. only disconnected nodes)
            if graph.number_of_edges() == 0:
                #print("is 0!")
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self.in_neigh_feats).to(feat_dst)

            if self._aggre_type == 'mean':
                graph.srcdata['h'] = feat_src
                graph.update_all(concatenate_edge_feats, lambda node: {'neigh': node.mailbox['m'].mean(axis=1)}) #message-passing: concatenate_edge_feats is applied on each edge, then the elementwise mean is computed
                h_neigh = graph.dstdata['neigh']#tensor that stores the aggregated features

            elif self._aggre_type == 'gcn':

                graph.srcdata['h'] = feat_src
                graph.dstdata['h'] = feat_dst
                graph.update_all(concatenate_edge_feats, lambda node: {'neigh': node.mailbox['m'].sum(axis=1)})
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)

            elif self._aggre_type == 'maxpool':
                if self.fc_pool is not None:
                    graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))#extra step: vertex features -> nn -> transformed features
                else:
                    graph.srcdata['h'] = feat_src

                graph.update_all(concatenate_edge_feats, lambda node: {'neigh': node.mailbox['m'].max(axis=1)}) #elementwise max
                h_neigh = graph.dstdata['neigh']

            elif self._aggre_type == 'meanpool':
                if self.fc_pool is not None:
                    #print("is cuda: ", h.is_cuda)
                    graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))#pooling over neighbours
                else:
                    graph.srcdata['h'] = feat_src

                graph.update_all(concatenate_edge_feats, lambda node: {'neigh': node.mailbox['m'].mean(axis=1)})
                h_neigh = graph.dstdata['neigh']

            elif self._aggre_type == 'lstm':
                graph.srcdata['h'] = feat_src
                #nf.layers[layer_id].data['h'] = h
                graph.update_all(concatenate_edge_feats, self._lstm_reducer)
                h_neigh = graph.dstdata['neigh']
                #nf.block_compute(layer_id, concatenate_edge_feats, self._lstm_reducer)
                #h_neigh = nf.layers[layer_id+1].data.pop('neigh')

            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))
            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == 'gcn':
                rst = self.fc_neigh(h_neigh)
            else:
                #if self.fc_self is not None:#fully connected over self + vector aggregated
                #print(h_self.shape)
                #print(h_neigh.shape)
                #print(torch.cat((h_self, h_neigh), 1).shape)
                rst = self.fc_neigh(torch.cat((h_self, h_neigh), 1))#concat and nn layer
            #else:
            #rst = self.fc_neigh(h_neigh)
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)

            return rst
