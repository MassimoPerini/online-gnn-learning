from collections import defaultdict
import numpy as np
import pandas as pd
import os
import json
import networkx as nx
from networkx.readwrite import json_graph
import utils

from graph.dynamic_graph_edge import DynamicGraphEdge
from ast import literal_eval as make_tuple

import dgl

#path = "datasets/reddit/"
FILES = ["feat_data.npy", "targets.npy", "edges_dataframe.csv"]
URL = "https://uoe-my.sharepoint.com/:u:/g/personal/s2121589_ed_ac_uk/EX6tn7RXc39LoIwZ0D5F9EcBofEGksT7nIuOrwIqCfXnPw?Download=1"

def preprocess(path, restrict=100000):

    edges_timestamps = json.load(open(os.path.join(path, "edge_timestamps.json")))
    id_map = json.load(open(os.path.join(path, "reddit-id_map.json")))
    G_final = nx.Graph()
    edges_timestamps_res = {}
    edges_timestamps_fixed = {}
    i=0

    G_data_json = json.load(open(os.path.join(path, "reddit-G.json")))
    G_data = json_graph.node_link_graph(G_data_json)

    max_len = len(G_data.edges())

    for edge in list(G_data.edges()):
        node_1 = edge[0] # get numerical vertex id
        node_2 = edge[1]

        key_1 = G_data_json["nodes"][node_1]["id"] #integer id -> string id
        key_2 = G_data_json["nodes"][node_2]["id"]

        #query the map
        intersection = set(edges_timestamps[key_1].keys()) & set(edges_timestamps[key_2].keys()) #check: map[key] : users that commented the post. check intersection (same user)
        min_time = -1
        first = True

        for k in intersection: #users who commented both posts
            time_1 = edges_timestamps[key_1][k]
            time_2 = edges_timestamps[key_2][k]
            time_edge = max(time_1, time_2)
            if (first or time_edge < min_time) and k != "":
                min_time = time_edge
                first = False
                edges_timestamps_res[str(edge)] = min_time #key: edge (vertex ids)
                edges_timestamps_fixed[str((id_map[key_1], id_map[key_2]))] = min_time
                G_final.add_edge(id_map[key_1], id_map[key_2])

        if i%1000 == 0:
            print(i, " of ", max_len)

        i+=1

    G_final = G_final.to_undirected()
    nx.write_adjlist(G_final, os.path.join(path, "graph.adjlist"))
    feat_data = np.load(os.path.join(path, "reddit-feats.npy"))

    js = json.dumps(edges_timestamps_res)
    f = open(os.path.join(path, "edge_timestamp_old.json"), "w")
    f.write(js)
    f.close()

    js = json.dumps(edges_timestamps_fixed)
    f = open(os.path.join(path, "edge_timestamp.json"), "w")
    f.write(js)
    f.close()

    np.save(os.path.join(path, "feat_data.npy"), feat_data.astype(np.double, order='C'), allow_pickle=False,
            fix_imports=True)

    labels = np.empty((feat_data.shape[0], 1), dtype=np.int64)
    targets_json = json.load(open(os.path.join(path, "reddit-class_map.json")))
    for k, v in targets_json.items():
        labels[id_map[k]] = int(v)

    np.save(os.path.join(path, "targets.npy"), feat_data.astype(np.double, order='C'), allow_pickle=False,
            fix_imports=True)


def relabel():
    feat_data = np.load("feat_data.npy")
    targets = np.load("targets.npy")
    G = nx.read_adjlist("graph.adjlist", nodetype=int)
    with open('edge_timestamp.json') as f:
        timestamps = json.load(f, object_hook=lambda d: {tuple(map(int, make_tuple(k))): v for k, v in d.items()})

    def sort_second(val):
        return val[1]

    ordered_edges = list(timestamps.items())
    ordered_edges.sort(key=sort_second)
    edges = list(x[0] for x in ordered_edges)

    counter = 0
    vertices = {}

    for e in edges:
        v0 = e[0]
        v1 = e[1]

        if v0 not in vertices:
            vertices[v0] = counter
            counter += 1
        if v1 not in vertices:
            vertices[v1] = counter
            counter += 1

    new_graph = nx.relabel_nodes(G, vertices, copy=True)  # final graph
    new_timestamps = {}  # final timestamps
    for e in edges:
        new_timestamps[(vertices[e[0]], vertices[e[1]])] = timestamps[e]

    order = list(vertices.items())
    order.sort(key=sort_second)
    order = list(x[0] for x in order)

    feat_data_ordered = feat_data[order, :]
    targets_ordered = targets[order, :]

    str_new_timestamps = {}
    for x in new_timestamps.items():
        str_new_timestamps[str((x[0][0], x[0][1]))] = x[1]

    new_graph = new_graph.to_undirected()
    nx.write_adjlist(new_graph, "graph_relabel.adjlist")

    js = json.dumps(str_new_timestamps)
    f = open(("edge_relabel_timestamp.json"), "w")
    f.write(js)
    f.close()

    np.save(("feat_relabel_data.npy"), feat_data_ordered.astype(np.double, order='C'), allow_pickle=False,
            fix_imports=True)
    np.save(("targets_relabel.npy"), targets_ordered.astype(np.double, order='C'), allow_pickle=False, fix_imports=True)


def load(path, snapshots=100, cuda=False, copy_to_gpu = False):
    a_exist = [f for f in FILES if os.path.isfile(os.path.join(path, f))]
    if len(a_exist) < len(FILES):
        from dataset_utils.common_utils import downloadFromURL
        downloadFromURL(URL, path, True)

    feat_data = np.load(os.path.join(path, "feat_data.npy"))
    targets = np.load(os.path.join(path, "targets.npy"))
    targets = targets.astype(np.long)

    timestamps = pd.read_csv(os.path.join(path, "edges_dataframe.csv"), na_filter=False, dtype=np.int)

    print(feat_data.shape)
    print(len(timestamps))

    feat_data_size = feat_data.shape[1]
    #if cuda and copy_to_gpu:
    #    print("copy to gpu")
    feat_data = utils.to_nn_lib(feat_data, False)
    labelled_vertices = set(np.argwhere(targets != -1)[:,0])
    n_classes = len(np.unique(targets))
    targets = utils.to_nn_lib(targets, False)


    dynamic_graph = DynamicGraphEdge(snapshots, labelled_vertices)
    dynamic_graph.build(feat_data, targets, cuda and copy_to_gpu, edge_timestamps=timestamps)

    print("DONE")
    #dynamic_graph_test = None

    dynamic_graph_test = DynamicGraphEdge(snapshots, labelled_vertices)
    dynamic_graph_test.build(feat_data, targets, cuda and copy_to_gpu, edge_timestamps=timestamps)

    return feat_data_size, targets, dynamic_graph, n_classes, dynamic_graph_test


if __name__ == "__main__":
    preprocess("../datasets/reddit")
