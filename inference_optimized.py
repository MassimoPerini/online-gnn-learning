print("xxxx", flush=True)
print("help", flush=True)

from torch import nn
from torch.nn import functional as F

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

print("IMPORTED", flush=True)
print("IMPORTED", file=sys.stderr)

projected_feat = ""

def concatenate_edge_feats(edges):
    return {'m': edges.src[projected_feat]}

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
        print("init start", flush=True)
        properties = ctx.system_properties
        #self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        model_dir = properties.get("model_dir")

        #####EDIT HERE
        ######################
        print("load dataset...", flush=True)
        dataset = "reddit"
        print(os.path.join(model_dir, dataset + ".json"), flush=True)
        with open(os.path.join(model_dir, dataset + ".json")) as settings:
            data = json.load(settings)
            print("loaded", flush=True)
            data["cuda"] = False
            data["dataset"] = "reddit"
            data["gpu"] = -1
            data["copy_dataset_gpu"] = False
            print(data["path"], flush=True)
            ########

        self.device = torch.device("cpu")
        if data["cuda"]:
            self.device = torch.device("cuda:0")

        print("settings loaded...", flush=True)
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

        print("dataset imported", flush=True)

        GraphSAGE, _, _, _, FullSupervisedGraphSage, activation = init(LIBRARY, data["cuda"], data["gpu"])
        print("init success", flush=True)
        #try:
        feat_data_size, self.labels, self.graph_feat, n_classes, _ = load("/project/graph-rl/priority_based_sampling/datasets/reddit", snapshots=1, cuda=data["cuda"], copy_to_gpu=data["copy_dataset_gpu"])
        print("loaded", self.graph_feat, flush=True)
        self.graph_feat = self.graph_feat.get_graph()#graph_feat.graph REDDIT
        print(self.graph_feat, flush=True)
        print("graph feat", flush=True)
        #except Exception as e:
        #    print(e)
        print("dataset loaded", flush=True)
        ######################

        # Read model serialize/pt file
        model_pt_path = os.path.join(model_dir, "reddit.pt")################
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
        #####
        self.fc_pool = []
        self.fc_neigh = []
        self.fc_self = []
        for x in range(2):#0: first layer
            layer = self.model.layers[x]#sageconv
            self.fc_pool += [layer.fc_pool]
            self.fc_neigh += [layer.fc_neigh]
            self.fc_self += [layer.fc_self]
        #####
        self.initialized = True
        print("INIT OK", flush=True)

    def inference(self, data):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # Convert 2D image to 1D vector
        output_data = []
        value = data[0].get("body")
        #print("RECEIVED ", value, "SIZE: ", len(self.graph), " ", self.requests)
        #start = time.time()
        value = json.loads(str(value))

        vertices = set()
        vertices_total = set()
        
        for sublist in value:
            vertices.add(sublist[0])
            vertices_total.add(sublist[0])
            vertices_total.add(sublist[1])

        v_copy = vertices.copy()
        #vertices_subgraph = [v_copy]
        
        #print("add vertices", flush=True)
        n_add_vertices = max(vertices_total)+1-len(self.graph)
        if n_add_vertices > 0:
            prev_size = len(self.graph)
            self.graph.add_nodes(n_add_vertices, {'h0proj': torch.zeros(n_add_vertices, self.fc_pool[0].out_features), 'h1proj': torch.zeros(n_add_vertices, self.fc_pool[1].out_features), 'h1': torch.zeros(n_add_vertices, self.fc_self[0].out_features), 'h2': torch.zeros(n_add_vertices, self.fc_self[1].out_features), 'neigh0': torch.zeros(n_add_vertices, self.fc_pool[0].out_features), 'neigh1': torch.zeros(n_add_vertices, self.fc_pool[1].out_features)})
            #print("copy feat n. vertices: ", n_add_vertices, flush=True)
            #print(self.graph.ndata, flush=True)
            #####TODO FIX HERE
            if 'feat' in self.graph.ndata:
                self.graph.ndata['feat'][prev_size:prev_size+n_add_vertices] = self.graph_feat.ndata['feat'][prev_size:prev_size+n_add_vertices]
                self.graph.ndata['target'][prev_size:prev_size+n_add_vertices] = self.graph_feat.ndata['target'][prev_size:prev_size+n_add_vertices]
            else:
                #print("GRAPH_FEAT: ",self.graph_feat.ndata['feat'], prev_size,"  ",prev_size+n_add_vertices)
                self.graph.ndata['feat'] = torch.tensor(self.graph_feat.ndata['feat'][prev_size:prev_size+n_add_vertices])
                self.graph.ndata['target'] = torch.tensor(self.graph_feat.ndata['target'][prev_size:prev_size+n_add_vertices])

        #print("SHAPE: ", self.graph.ndata['feat'].shape, flush=True)
        #print("vertices ", len(self.graph), flush=True)
        for edge in value:
            #print(edge)
            self.graph.add_edge(edge[1], edge[0])#reverse
        #return []
        #for _ in range(1):
        #print("vertices loop ", vertices)

        #vertices_subgraph += [list(vertices)]
        sampling_th = 15
        l_vertices = np.array(list(vertices))    
        ten_vertices = torch.from_numpy(l_vertices).type(torch.LongTensor)
        if self.copy_dataset_gpu:
            ten_vertices = ten_vertices.cuda()
        
        out_degrees = self.graph.out_degrees(ten_vertices).flatten().numpy()
        idx_keep = np.where(out_degrees < sampling_th)[0]
        l_vertices = l_vertices[idx_keep]
        vertices_subgraph = [set(l_vertices.tolist())]
        ten_vertices = torch.from_numpy(l_vertices).type(torch.LongTensor)
        
        if self.copy_dataset_gpu:
            ten_vertices = ten_vertices.cuda()

        _, succs = self.graph.out_edges(ten_vertices)
        succs = succs.flatten()
        pred, _ = self.graph.in_edges(ten_vertices)
        #print("succs: ", _, succs, succs.dim())
        #succs = succs.flatten().tolist()
        pred = pred.flatten().tolist()
        #vertices = vertices.union(set(succs))
        #v_copy = vertices.copy()
        #vertices_subgraph += [list(v_copy)]
        vertices_subgraph += [list(set(pred))]
        
        out_degrees = self.graph.out_degrees(succs).flatten().numpy()
        idx_keep = np.where(out_degrees < sampling_th)[0]
        print("IDX_KEEP ", idx_keep)
        succs = succs.numpy()[idx_keep]
        succs = succs.tolist()
        
        vertices_subgraph += [list(set(succs))]

        #print("vertices ", vertices)
        #train_vertices = torch.LongTensor(list(vertices))
        num_workers = 0
        batch_size = 32
        #print("SHAPE: ", self.graph.ndata['feat'].shape, flush=True)
        #print("vertices ", len(self.graph), flush=True)
        #print(self.graph.ndata)
        #print(self.graph.node_attr_schemes())
        #print(self.graph.edata)
        #print("train_vertices ", train_vertices)
        #vv = []
        #vv += [list(vertices_subgraph[0])]
        #vv += [list(vertices_subgraph[1])]
        #vertices_subgraph = vv
        ###
        #train_subgraph_1 = self.graph.reverse(share_ndata=True, share_edata=True)
        start = time.time()
        for i in range(2):
            #nids = list(vertices_subgraph[i])
            if i == 0:
                nids = list(vertices_subgraph[i])
                h_self = self.graph.ndata['feat'][nids]#train_subgraph_1.ndata['feat']
            else:
                nids = list(vertices_subgraph[i+1])
                h_self = self.graph.ndata['h'+str(i)][nids]#train_subgraph_1.ndata['h'+str(i)]
            if self.cuda and not self.copy_dataset_gpu:
                h_self = h_self.cuda()
            
            #generate proj for new vertices
            global projected_feat
            projected_feat = 'h'+str(i)+"proj"
            #print(F.relu(self.fc_pool[i](h_self)))
            if self.cuda and not self.copy_dataset_gpu:
                self.graph.ndata[projected_feat][nids] = F.relu(self.fc_pool[i](h_self)).cpu()
            else:
                self.graph.ndata[projected_feat][nids] = F.relu(self.fc_pool[i](h_self))
            #print(self.graph.ndata[projected_feat][nids])
            #mean over neighbours
            #nids2=list(vertices_subgraph[i+1])
            train_subgraph_2 = self.graph.subgraph(vertices_subgraph[i+1])
            if self.cuda and not self.copy_dataset_gpu:
                train_subgraph_2 = train_subgraph_2.to('cuda:0')
            #train_subgraph_2 = train_subgraph_2.reverse(share_ndata=True, share_edata=True)
            train_subgraph_2.update_all(concatenate_edge_feats, lambda node: {'neigh'+str(i): node.mailbox['m'].mean(axis=1)})
            h_neigh = train_subgraph_2.dstdata['neigh'+str(i)]
            #allocate neigh
            if self.cuda and not self.copy_dataset_gpu:
                self.graph.ndata['neigh'+str(i)][train_subgraph_2.ndata[dgl.NID]] = h_neigh.cpu()
                rst = self.fc_self[i](h_self) + self.fc_neigh[i](self.graph.ndata['neigh'+str(i)][nids].cuda())
            else:
                self.graph.ndata['neigh'+str(i)][train_subgraph_2.ndata[dgl.NID]] = h_neigh
                rst = self.fc_self[i](h_self) + self.fc_neigh[i](self.graph.ndata['neigh'+str(i)][nids])#fc_neigh[i](torch.cat((h_self, h_neigh), 1))
            if i < 1:
                rst = torch.nn.functional.relu(rst)
            if self.cuda and not self.copy_dataset_gpu:
                self.graph.ndata['h' + str(i+1)][nids] = rst.cpu()
            else:
                self.graph.ndata['h' + str(i+1)][nids] = rst

        #print(self.graph.ndata['h2'][vertices_subgraph[-2]])
        end = time.time()
        values, indices = (self.graph.ndata['h2'][vertices_subgraph[-2]]).max(1)
        if self.cuda:
            output_data = indices.data.cpu().tolist()
        else:
            output_data = indices.tolist()
        #end = time.time()
        self.requests += 1
        comp_time = end - start
        #print("COMP TIME ",comp_time)
        self.file_results.write(str(comp_time)+"\n")
        print("RECEIVED ", value, "SIZE: ", len(self.graph), " ", self.requests, " ", comp_time)
        if self.requests % 2500 == 0:#REDDIT
            self.file_results.close()
            print("SAVE", flush=True)
            self.file_results = open("/project/graph-rl/priority_based_sampling/results_inf.txt", "a")
        return [str(output_data)]

_service = MNISTDigitClassifier()


def handle(data, context):
    #f=open("/project/graph-rl/priority_based_sampling/log.txt", "a+")
    #f.write("xxxx")
    #f.close()
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.inference(data)

    return data
