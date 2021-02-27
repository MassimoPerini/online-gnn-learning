import numpy as np
import pandas as pd
import networkx as nx
import os
import json

import dgl
import torch
import utils

from graph.dynamic_graph_vertex import DynamicGraphVertex

path = "datasets/bitcoin/"
FILES = ["feat_data.npy", "targets.npy", "graph.adjlist", "vertex_timestamp.json"]
URL = "https://file.perini.me/graphs/bitcoin.zip"

def preprocess(path):
    df1 = pd.read_csv("elliptic_txs_classes.csv")
    df2 = pd.read_csv("elliptic_txs_features.csv", header=None)
    df3 = pd.read_csv("elliptic_txs_edgelist.csv")
    df1 = df1[df1["class"] != "unknown"]

    node_ids = set(df3["txId1"].unique())
    node_ids = node_ids.union(set(df3["txId2"]))
    d={x:i for i,x in enumerate(node_ids)}

    i = 0
    for key, value in d.items():
        df3.loc[df3['txId1'] == key, ['txId1']] = value
        df3.loc[df3['txId2'] == key, ['txId2']] = value
        i += 1
        if i%100 == 0:
            print (i)

    graph = nx.from_pandas_edgelist(df3, source='txId1', target='txId2', edge_attr=None, create_using=None)
    graph = graph.to_undirected()
    nx.write_adjlist(graph, "graph.adjlist")
    targets = np.zeros((len(d), 1), dtype=np.int64)

    i = 0
    success = 0
    for key, value in d.items():
        target = df1[df1['txId'] == key]["class"].values
        if len(target) > 0:
            target = int(target[0])
            targets[value] = target
            success += 1
        i += 1
        if i%100 == 0:
            print (i)
            print(success)

    targets[targets == 0] = -1
    np.save("targets.npy", targets.astype(np.double, order='C'), allow_pickle=False, fix_imports=True)
    feats = np.zeros((len(d), 165))

    i = 0
    for key, value in d.items():
        row = (df2[df2[0] == key].values)[0][2:]
        feats[value] = row
        i += 1
        if i%100 == 0:
            print (i)

    np.save("feat_data.npy", feats.astype(np.double, order='C'), allow_pickle=False, fix_imports=True)
    timestamps = {}
    i = 0
    for key, value in d.items():
        time = (df2[df2[0] == key].values)[0][1]
        timestamps[value] = time
        i += 1
        if i%100 == 0:
            print (i)

    with open('vertex_timestamp_last.json', 'w') as fp:
        json.dump(timestamps, fp)

def load(path, snapshots=100, cuda=False, copy_to_gpu=False):

    a_exist = [f for f in FILES if os.path.isfile(os.path.join(path, f))]
    if len(a_exist) < len(FILES):
        from dataset_utils.common_utils import downloadFromURL
        downloadFromURL(URL, path, True)

    feat_data = np.load(os.path.join(path, "feat_data.npy"))
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

if __name__ == "__main__":
    preprocess("../datasets/bitcoin")
