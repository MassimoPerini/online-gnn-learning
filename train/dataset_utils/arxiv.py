import numpy as np
import pandas as pd
import networkx as nx
import os
import json
import dgl
import torch
import utils

from graph.dynamic_graph_vertex import DynamicGraphVertex

path = "datasets/arxiv/"


FILES = ["feats.npy", "targets.npy", "graph.adjlist", "vertex_timestamp.json"]
URL = "todo"

def load(path, snapshots=100, cuda=False, copy_to_gpu=False):
    
    a_exist = [f for f in FILES if os.path.isfile(os.path.join(path, f))]
    if len(a_exist) < len(FILES):
        from dataset_utils.common_utils import downloadFromURL
        downloadFromURL(URL, path, True)

    feat_data = np.load(os.path.join(path, "feats.npy"))
    targets = np.load(os.path.join(path, "targets.npy"))
    targets = targets.astype(np.long)
    G = nx.read_adjlist(os.path.join(path, "graph.adjlist"), nodetype=int)

    with open(os.path.join(path, 'vertex_timestamp.json')) as f:#last
        timestamps = json.load(f, object_hook=lambda d: {int(k): v for k, v in d.items()})

    g_dgl = dgl.from_networkx(G)
    g_dgl_test = dgl.from_networkx(G)

    feat_data_size = feat_data.shape[1]
    feat_data = utils.to_nn_lib(feat_data, cuda and copy_to_gpu)
    labelled_vertices = set(np.argwhere(targets != -1)[:, 0])
    n_classes =	len(np.unique(targets))
    targets = utils.to_nn_lib(targets, cuda and copy_to_gpu)

    g_dgl.ndata['feat'] = feat_data
    g_dgl.ndata['target'] = targets

    g_dgl_test.ndata['feat'] = feat_data
    g_dgl_test.ndata['target'] = targets

    dynamic_graph = DynamicGraphVertex(g_dgl, snapshots, labelled_vertices)
    dynamic_graph.build(vertex_timestamps=timestamps)

    dynamic_graph_test = DynamicGraphVertex(g_dgl_test, snapshots, labelled_vertices)
    dynamic_graph_test.build(vertex_timestamps=timestamps)

    return feat_data_size, targets, dynamic_graph, n_classes, dynamic_graph_test
