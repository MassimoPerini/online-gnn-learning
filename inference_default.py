print("xxxx", flush=True)
print("help", flush=True)


import io
import os
#import logging
import torch
import numpy as np
import json
from torch.autograd import Variable
from torchvision import transforms
import dgl
import json
import sys
import time

print("Starting inf...", flush=True)
from utils import Lib_supported
LIBRARY = Lib_supported.PYTORCH

print("preload ok", flush=True)

class MNISTDigitClassifier(object):
    """
    MNISTDigitClassifier handler class. This handler takes a greyscale image
    and returns the digit in that image.
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.requests = 0
        self.file_results = open("/project/graph-rl/priority_based_sampling/results_inf.txt", "a")

    def initialize(self, ctx):
        """First try to load torchscript else load eager mode state_dict based model"""
        print("init start")
        f=open("/project/graph-rl/priority_based_sampling/log.txt", "a+")
        f.write("init start")
        f.close()
        properties = ctx.system_properties
        #self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        model_dir = properties.get("model_dir")

        #####EDIT HERE
        ######################
        print("load dataset...")
        dataset = "reddit"
        print(os.path.join(model_dir, dataset + ".json"))
        with open(os.path.join(model_dir, dataset + ".json")) as settings:
            data = json.load(settings)
            print("loaded")
            data["cuda"] = False
            data["dataset"] = "reddit"
            data["gpu"] = -1
            data["copy_dataset_gpu"] = False
            print(data["path"])
            ########

        self.device = torch.device("cpu")
        if data["cuda"]:
            self.device = torch.device("cuda:0")

        print("settings loaded...")
        from utils import init
        if data["dataset"] == "pubmed":
            from dataset_utils.pubmed import load
            print("imported pubmed")
        elif data["dataset"] == "reddit":
            from dataset_utils.reddit import load
        elif data["dataset"] == "elliptic":
            from dataset_utils.bitcoin import load
        elif data["dataset"] == "tagged":
            from dataset_utils.tagged import load
        elif data["dataset"] == "flickr":
            from dataset_utils.flickr import load
        elif data["dataset"] == "arxiv":
            from dataset_utils.arxiv import load

        print("dataset imported")

        GraphSAGE, _, _, _, FullSupervisedGraphSage, activation = init(LIBRARY, data["cuda"], data["gpu"])
        print("init success")
        #try:
        feat_data_size, self.labels, self.graph_feat, n_classes, _ = load("/project/graph-rl/priority_based_sampling/datasets/reddit", snapshots=1, cuda=data["cuda"], copy_to_gpu=data["copy_dataset_gpu"])
        print("loaded")
        self.graph_feat = self.graph_feat.get_graph()#graph_feat.graph REDDIT#self.graph_feat = self.graph_feat.graph
        print("graph feat")
        #except Exception as e:
        #    print(e)
        print("dataset loaded")
        ######################

        # Read model serialize/pt file
        model_pt_path = os.path.join(model_dir, "reddit.pt")#model_pt_path = os.path.join(model_dir, "gnn.pt")################
        # Read model definition file
        #model_def_path = "train/graphsage/pytorch/graphsage_dgl.py"#todo load from path
        print("read model def", flush=True)
        model_def_path = os.path.join(model_dir, "graphsage_dgl.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model definition file")

        print("loading weights", flush=True)
        state_dict = torch.load(model_pt_path, map_location=self.device)
        print("loading model", flush=True)
        ###########################
        self.model = GraphSAGE(feat_data_size, data["embedding_size"], n_classes, data["depth"] - 1, activation, data["dropout"], "pool", edge_feats=data["edge_feats"], pool_feats=data["latent_dim"])
        ###########################

        self.model.load_state_dict(state_dict)
        if data["cuda"]:
            self.model = self.model.cuda()
        self.model.eval()
        print("model loaded", flush=True)
        self.graph = dgl.DGLGraph()
        if data["cuda"] and data["copy_dataset_gpu"]:
            self.graph = self.graph.to('cuda:0')
        self.cuda = data["cuda"]
        self.copy_dataset_gpu = data["copy_dataset_gpu"]
        self.initialized = True
        print("INIT OK", flush=True)

    def inference(self, data):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # Convert 2D image to 1D vector
        #return []
        output_data = []
        value = data[0].get("body")
        #print(value, flush=True)
        value = json.loads(str(value))
        #start = time.time()
        vertices = set()
        vertices_total = set()
        for sublist in value:
            vertices.add(sublist[0])
            vertices_total.add(sublist[0])
            vertices_total.add(sublist[1])
        
        #print("add vertices", flush=True)
        n_add_vertices = max(vertices_total)+1-len(self.graph)
        if n_add_vertices > 0:
            prev_size = len(self.graph)
            self.graph.add_nodes(n_add_vertices)
            #print("copy feat n. vertices: ", n_add_vertices, flush=True)
            #print(self.graph.ndata, flush=True)
            if 'feat' in self.graph.ndata:
                self.graph.ndata['feat'][prev_size:prev_size+n_add_vertices] = self.graph_feat.ndata['feat'][prev_size:prev_size+n_add_vertices]
                self.graph.ndata['target'][prev_size:prev_size+n_add_vertices] = self.graph_feat.ndata['target'][prev_size:prev_size+n_add_vertices]
            else:
                self.graph.ndata['feat'] = torch.tensor(self.graph_feat.ndata['feat'][prev_size:prev_size+n_add_vertices])
                self.graph.ndata['target'] = torch.tensor(self.graph_feat.ndata['target'][prev_size:prev_size+n_add_vertices])

        #print("SHAPE: ", self.graph.ndata['feat'].shape, flush=True)
        #print("vertices ", len(self.graph), flush=True)
        for edge in value:
            #print(edge)
            self.graph.add_edge(edge[0], edge[1])
        #return []
        for _ in range(1):
            #print("vertices loop ", vertices)
            l_vertices = list(vertices)
            ten_vertices = torch.LongTensor(l_vertices)
            if self.copy_dataset_gpu:
                ten_vertices = ten_vertices.cuda()
            
            sample_degree = 15
            edited = False
            degrees = self.graph.out_degrees(ten_vertices).flatten().tolist()
            for i in range(len(degrees)):
                if degrees[i] > sample_degree:
                    edited = True
                    vertices.remove(l_vertices[i])
            if edited:
                ten_vertices = torch.LongTensor(list(vertices))
            
            succs, _ = self.graph.in_edges(ten_vertices)
            degrees = self.graph.out_degrees(succs).flatten().tolist()
            succs_list = succs.flatten().tolist()
            for i in range(len(degrees)):
                if degrees[i] < sample_degree:
                    vertices.add(succs_list[i])
            
            #print("succs[0]...1", succs[0], succs[1], succs[1].dim())
            #if succs[1].dim() == 0:
            #    vertices.add(succs[1].item())
            #else:
            #    vertices = vertices.union(set(succs[1]))

        print("neighbors ", vertices)
        train_vertices = torch.LongTensor(list(vertices))
        num_workers = 0
        batch_size = 32
        #print("SHAPE: ", self.graph.ndata['feat'].shape, flush=True)
        #print("vertices ", len(self.graph), flush=True)
        #print(self.graph.ndata)
        #print(self.graph.node_attr_schemes())
        #print(self.graph.edata)
        #print("train_vertices ", train_vertices)
        start=time.time()
        sampler = dgl.sampling.MultiLayerNeighborSampler([None for x in range(2)], replace=True)
        dataloader = dgl.sampling.NodeDataLoader(self.graph, train_vertices, sampler, batch_size=batch_size, shuffle=False, drop_last=False, num_workers = 0)
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            #print("processing ", seeds, input_nodes, flush=True)
            batch_inputs = self.graph.ndata['feat'][input_nodes]
            if self.cuda and not self.copy_dataset_gpu:
                batch_inputs = batch_inputs.cuda()
                blocks = [block.to(torch.device("cuda")) for block in blocks]
            val_output = self.model(blocks, batch_inputs)
            values, indices = val_output.max(1)
            if self.cuda:
                output_data += [indices.data.cpu().numpy()]
            else:
                output_data += [indices.numpy()]
            #print("output data: ", output_data, flush=True)
        comp_time=time.time()-start
        self.requests += 1
        self.file_results.write(str(comp_time)+"\n")
        print("RECEIVED ", value, "SIZE: ", len(self.graph), " ", self.requests, " ", comp_time)
        if self.requests % 2500 == 0:#REDDIT
            self.file_results.close()
            print("SAVE", flush=True)
            self.file_results = open("/project/graph-rl/priority_based_sampling/results_inf.txt", "a")

        return [str(list(np.concatenate(output_data)))]

_service = MNISTDigitClassifier()


def handle(data, context):
    f=open("/project/graph-rl/priority_based_sampling/log.txt", "a+")
    f.write("xxxx")
    f.close()
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.inference(data)

    return data
