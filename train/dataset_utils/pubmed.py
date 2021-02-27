print("imported pubmed loader")

from collections import defaultdict
import numpy as np
import os
import networkx as nx
import json

from graph.dynamic_graph_vertex import DynamicGraphVertex
import dgl

import utils

#path = "datasets/pubmed/"
FILES = ["feat_data.npy", "targets.npy", "graph.adjlist", "postponed_timestamp.json"]
URL = "https://file.perini.me/graphs/pubmed.zip"

def preprocess(path):
    # hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open(os.path.join(path, "Pubmed-Diabetes.NODE.paper.tab")) as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1]) - 1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])

    np.save(os.path.join(path, "feat_data.npy"), feat_data.astype(np.double, order='C'), allow_pickle=False,
            fix_imports=True)
    np.save(os.path.join(path, "targets.npy"), labels.astype(np.double, order='C'), allow_pickle=False,
            fix_imports=True)

    G = nx.Graph()

    adj_lists = defaultdict(set)
    with open(os.path.join(path, "Pubmed-Diabetes.DIRECTED.cites.tab")) as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            #adj_lists[paper1].add(paper2)
            #adj_lists[paper2].add(paper1)
            G.add_edge(paper1, paper2)

    G = G.to_undirected()
    nx.write_adjlist(G, os.path.join(path, "graph.adjlist"))

    remapped_timestamps = {}

    with open(os.path.join(path, 'pubmed-timestamp_map.json')) as f:
        d = json.load(f)
        for key, value in d.items():
            remapped_timestamps[int(node_map[key])] = float(value)

    js = json.dumps(remapped_timestamps)
    f = open(os.path.join(path, "vertex_timestamp.json"), "w")
    f.write(js)
    f.close()

def load(path, snapshots=100, cuda=False, copy_to_gpu = False):
    print("loading data...")
    a_exist = [f for f in FILES if os.path.isfile(os.path.join(path, f))]
    if len(a_exist) < len(FILES):
        from dataset_utils.common_utils import downloadFromURL
        downloadFromURL(URL, path, True)

    feat_data = np.load(os.path.join(path, "feat_data.npy"))
    targets = np.load(os.path.join(path, "targets.npy"))
    targets = targets.astype(np.long)
    G = nx.read_adjlist(os.path.join(path, "graph.adjlist"), nodetype = int)

    with open(os.path.join(path, 'postponed_timestamp.json')) as f: #vertex
        timestamps = json.load(f, object_hook= lambda d: {int(k):v for k, v in d.items()})

    g_dgl = dgl.from_networkx(G)
    g_dgl_test = dgl.from_networkx(G)

    feat_data_size = feat_data.shape[1]

    print("moving to nn lib")

    #print(np.any(np.isnan(feat_data)))

    feat_data = utils.to_nn_lib(feat_data, cuda and copy_to_gpu)

    #import torch

    #print(torch.isnan(feat_data).any())

    labelled_vertices = set(np.argwhere(targets != -1)[:,0])
    n_classes =	len(np.unique(targets))
    targets = utils.to_nn_lib(targets, cuda and copy_to_gpu)

    if cuda and copy_to_gpu:
        g_dgl = g_dgl.to('cuda:0')
        g_dgl_test = g_dgl_test.to('cuda:0')

    g_dgl.ndata['feat'] = feat_data
    g_dgl.ndata['target'] = targets

    #print(torch.isnan(g_dgl.ndata['feat']).any())

    g_dgl_test.ndata['feat'] = feat_data
    g_dgl_test.ndata['target'] = targets

    dynamic_graph = DynamicGraphVertex(g_dgl, snapshots, labelled_vertices)
    dynamic_graph.build(vertex_timestamps=timestamps)

    dynamic_graph_test = DynamicGraphVertex(g_dgl_test, snapshots, labelled_vertices)
    dynamic_graph_test.build(vertex_timestamps=timestamps)

    print("loaded")

    return feat_data_size, targets, dynamic_graph, n_classes, dynamic_graph_test


if __name__ == "__main__":
    print("main")
    preprocess("datasets/pubmed")
