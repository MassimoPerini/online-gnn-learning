from graphsage.tf_static.aggregator_dgl import SAGEConv
from tensorflow.keras import Model
import tensorflow as tf

class GraphSAGE(Model):#todo
    '''
    Defines the GraphSAGE model
    '''
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,
                 edge_feats = None,
                 pool_feats = None):
        '''
        Init the NN
        :param in_feats: size of the vertex features
        :param n_hidden: hidden dimension neurons
        :param n_classes: N. classes (vertex)
        :param n_layers: n. of convolutional layers (sampling depth)
        :param activation: activation function
        :param dropout: dropout
        :param aggregator_type: type of aggregation (mean, meanpool, maxpool, ...)
        :param edge_feats: size of the features on the edges (or None if the graph has no edge features)
        :param pool_feats: size of the pooling dimension (only pooling aggregators)
        '''

        super(GraphSAGE, self).__init__()

        self._layers = []
        #self.g = g
        # input layer
        self._layers += [SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation, edge_feats=edge_feats, pool_feats=pool_feats)]
        # hidden layers
        for i in range(n_layers - 1):
            self._layers += [SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation, edge_feats=edge_feats, pool_feats=pool_feats)]
        # output layer
        self._layers += [SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None, edge_feats=edge_feats, pool_feats=pool_feats)] # activation None

    def call(self, tensors, matrices, real_batch):
        '''
        Apply NN Layers
        :param nf: instance of NeighborSampler: generates the k-hop sampled neighbourhood
        :return: vertex predictions
        '''

        for j, layer in enumerate(self._layers):  # "convolutional depth" step
            result = layer(tensors[j], tensors[j+1], matrices[j])
            for x in range(j+1, self.get_n_layers()):
                tensors[x] = result
                result = result[:real_batch[x+1]]
            tensors[-1] = result

        h = tensors[-1]
        return h

    def get_n_layers(self):
        return len(self._layers)