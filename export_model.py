import argparse
import json
import os
from prioritized_replay.generate_priority import *
import torch

START_PRIOR_ALPHA = 4#10#1#0.3
END_PRIOR_ALPHA = 50#80#4#14
SCALE = 1

parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=['elliptic', 'pubmed', 'reddit', 'tagged', 'flickr', 'arxiv'], help="Dataset")
parser.add_argument("backend", choices=['tf', 'pytorch', 'tf_static'], help="Framework (Tensorflow 2 or Pytorch)")
parser.add_argument("save_result", help="output file (.csv)")
parser.add_argument("save_tsne", help="path tsne plots")

parser.add_argument("--cuda", action='store_true', help="Enable CUDA")
parser.add_argument("--gpu", type=int, default=-1, help="Use a specific GPU (only if CUDA is enabled)")

parser.add_argument("--snapshots", type=int, help="split the temporal graph into N snapshots")
parser.add_argument("--embedding_size", type=int, help="intermediate latent size (between aggregation hops)")
parser.add_argument("--latent_dim", type=int, help="Pooling layers output size")
parser.add_argument("--depth", type=int, help="sampling depth")
parser.add_argument("--samples", type=int, help="neighbours sampled")
parser.add_argument("--batch_timestep", type=int, help="train N batches in every snapshot")
parser.add_argument("--eval", type=int, help="evaluate the model every N snapshots")
parser.add_argument("--batch_size", type=int, help="batch size used during the training phase")
parser.add_argument("--batch_full", type=int, help="batch size used during the testing phase")
parser.add_argument("--epochs_offline", type=int, help="trains the offline model for N epochs")
parser.add_argument("--train_offline", type=int, help="trains the offline model every N snapshots")
parser.add_argument("--priority_forward", type=int, help="update the priorities running a forward pass every N snapshots")
parser.add_argument("--plot_tsne", type=int, help="generate a TSNE plot every N snapshots")
parser.add_argument("--dropout", type=float, help="dropout used during the training phase")
parser.add_argument("--delta", type=int, help="evaluate the current model over a graph N snapshot in the future")

parser.add_argument("--n_sampling_workers", type=int, help="n. of parallel workers that sample the graph", default=0)
parser.add_argument("--copy_dataset_gpu", action='store_true', help="Copy the dataset to the GPU memory. Otherwise each batch will be copied")


args = parser.parse_args()
print(args, flush=True)
custom_settings = {k: v for k, v in vars(args).items() if v is not None}

with open('settings/'+args.dataset+".json") as settings:
    data = json.load(settings)
    data.update(custom_settings)

if args.backend == "tf" or args.backend == "tf_static":
    os.environ['USE_OFFICIAL_TFDLPACK'] = "true"
    os.environ['DGLBACKEND'] = "tensorflow"
elif args.backend == "pytorch":
    os.environ['DGLBACKEND'] = "pytorch"

from utils import Lib_supported
if args.backend == "tf":
    LIBRARY = Lib_supported.TF
elif args.backend == "tf_static":
    LIBRARY = Lib_supported.TF_STATIC
elif args.backend == "pytorch":
    LIBRARY = Lib_supported.PYTORCH

import numpy as np
from graph.train_test_graph import TrainTestGraph
import random
from utils import init


import os

def run():
    print("init", flush=True)
    GraphSAGE, _, _, _, FullSupervisedGraphSage, activation = init(LIBRARY, data["cuda"], data["gpu"])
    if args.dataset == "pubmed":
        from dataset_utils.pubmed import load
    elif args.dataset == "reddit":
        from dataset_utils.reddit import load
    elif args.dataset == "elliptic":
        from dataset_utils.bitcoin import load
    elif args.dataset == "tagged":
        from dataset_utils.tagged import load
    elif args.dataset == "flickr":
        from dataset_utils.flickr import load
    elif args.dataset == "arxiv":
        from dataset_utils.arxiv import load

    print("load data", flush=True)
    feat_data_size, labels, graph, n_classes, _ = load(data["path"], snapshots=1, cuda=data["cuda"], copy_to_gpu = data["copy_dataset_gpu"])
    print("train test init", flush=True)
    graph_util = TrainTestGraph(graph, split=0.15, start_prior_alpha=START_PRIOR_ALPHA, end_prior_alpha=END_PRIOR_ALPHA, scale=SCALE, max_priority=10)
    print("create graphsage", flush=True)
    model_full = GraphSAGE(feat_data_size, data["embedding_size"], n_classes, data["depth"] - 1, activation, data["dropout"], "pool", edge_feats=data["edge_feats"], pool_feats=data["latent_dim"])
    if data["cuda"] and LIBRARY == Lib_supported.PYTORCH:
        print("moving models to CUDA... ", flush=True)
        model_full = model_full.cuda()

    graphsage_full = FullSupervisedGraphSage(model_full, data["epochs_offline"], data["batch_size"], labels, data["samples"], cuda=data["cuda"],
                                             batch_full=data["batch_full"], n_workers = data["n_sampling_workers"])
    graphsage_full.build_optimizer()

    size_evolution = len(graph_util)
    print(size_evolution, flush=True)

    graphsage_full.train_timestep(graph_util)
    print("trained", flush=True)
    graphsage_full.evaluate(graph_util, data["save_result"])

    torch.save(model_full.state_dict(), "gnn.pt")
    #torch-model-archiver --model-name gnn --version 1.0 --model-file model.py --serialized-file gnn.pt --handler handler.py
    #torch-model-archiver --model-name gnn --version 1.0 --model-file train/graphsage/pytorch/graphsage_dgl.py --serialized-file gnn.pt --handler train/inference.py


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    np.random.seed(1)
    random.seed(1)
    run()

