import torch
import torch.nn as nn
import numpy as np
import dgl
import utils
from graphsage.model import SupervisedGraphSage

"""
Supervised GraphSAGE model
"""

class PytorchSupervisedGraphSage(SupervisedGraphSage):
    '''
    Several GraphSAGE train methods
    '''

    def __init__(self, graphsage_model, batch_per_timestep, batch_size, labels, samples, reduction='mean', n_workers = 1, cuda=False, batch_full = 512):
        super(PytorchSupervisedGraphSage, self).__init__(graphsage_model, batch_per_timestep, batch_size, labels, samples, n_workers, cuda, batch_full)
        #self.n_workers = 0
        self.xent = nn.CrossEntropyLoss(reduction=reduction)
        if cuda:
            self.xent = nn.CrossEntropyLoss(reduction=reduction).cuda()

    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.graphsage_model.parameters(), lr=0.001)

    def copy_edges_feats(self, graph, blocks):
        for block in blocks:
            if graph.edata['feat'].is_sparse:
                sparse_tensor = graph.edata['feat']
                idx = block.edata[dgl.EID]
                col_idx = sparse_tensor._indices()[1][idx]
                row_idx = torch.arange(len(idx))
                block.edata["feat"] = torch.sparse.LongTensor(torch.stack((row_idx, col_idx)), sparse_tensor._values()[idx], torch.Size([len(idx), sparse_tensor.shape[1]]))
            else:
                block.edata["feat"] = graph.edata['feat'][block.edata[dgl.EID]]


    def _run_custom_eval(self, graph, subgraph_to_id, id_to_subgraph, test_vertices):
        output_data = []
        self.graphsage_model.eval()
        batch_nodes_seed = torch.LongTensor(test_vertices)

        sampler = dgl.sampling.MultiLayerNeighborSampler([self.samples for x in range(2)], replace=True, return_eids=True)
        dataloader = dgl.sampling.NodeDataLoader(graph, batch_nodes_seed, sampler,
                                                 batch_size=self.batch_full,
                                                 shuffle=False, drop_last=False, num_workers=self.n_workers)

        #print("batch full", self.batch_full)
        import time
        #t_start = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            #print("t_yeld", time.time()-t_start)
            batch_inputs = graph.ndata['feat'][input_nodes]  # .to(device)

            if 'feat' in graph.edata:
                self.copy_edges_feats(graph, blocks)
            #t_cuda = time.time()

            if self._cuda_var:
                batch_inputs = batch_inputs.cuda()
                blocks = [block.to(torch.device("cuda")) for block in blocks]

            #print("cuda ", time.time()-t_cuda)
            #t_comp = time.time()
            val_output = self.graphsage_model(blocks, batch_inputs)
            output_data += [val_output.data.cpu().numpy()]
            #print("time computation ", time.time()-t_comp)
            #t_start = time.time()

        return output_data


    def get_model(self):
        return "base_model"

    def train_step(self, graph, blocks, input_nodes, seeds, subgraph_to_id):
        '''
        Computes the loss of the given NodeFlow
        :param nf: A NodeFlow
        :param subgraph_to_id_map: map subgraph id -> original graph id
        :return: Loss value
        '''

        #batch_nids = nf.layer_parent_nid(-1)
        #batch_nodes_seed = [subgraph_to_id_map[v.item()] for v in batch_nids]
        self.optimizer.zero_grad()
        batch_inputs = graph.ndata['feat'][input_nodes]  # .to(device)
        #print(torch.isnan(batch_inputs))
        #print(torch.isnan(batch_inputs).any())
        batch_labels = graph.ndata['target'][seeds] #self.labels[subgraph_to_id[seeds]]  # .to(device)

        if 'feat' in graph.edata:
            self.copy_edges_feats(graph, blocks)

        if self._cuda_var:
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.cuda()
            blocks = [block.to(torch.device("cuda")) for block in blocks]

        #batch_labels = torch.autograd.Variable(torch.LongTensor(batch_labels))
        #with torch.autograd.detect_anomaly():
        scores = self.graphsage_model(blocks, batch_inputs)

        loss = self.xent(scores, batch_labels.flatten())
        loss.backward()
        self.optimizer.step()


class RandomPytorchSupervisedGraphSage(PytorchSupervisedGraphSage):
    '''
    A GraphSAGE model trained on random vertices
    '''

    def __init__(self, model, batch_per_timestep, batch_size, labels, samples, cuda=False, batch_full=512, n_workers=0):
        super(RandomPytorchSupervisedGraphSage, self).__init__(model, batch_per_timestep, batch_size, labels, samples, n_workers=n_workers, cuda=cuda, batch_full = batch_full)

    def choose_vertices(self, graph_util):
        batch_nodes = []
        for xx in range(self.batch_per_timestep):
            batch_nodes += graph_util.draw_random_train_nodes(self.batch_size)
        return batch_nodes

    def _run_custom_train(self, graph, subgraph_to_id, id_to_subgraph, train_vertices, graph_util):
        self.graphsage_model.train()
        train_vertices = torch.LongTensor(train_vertices)

        sampler = dgl.sampling.MultiLayerNeighborSampler([self.samples for x in range(2)], replace=True, return_eids=True)
        dataloader = dgl.sampling.NodeDataLoader(graph, train_vertices, sampler,
            batch_size=len(train_vertices)//self.batch_per_timestep,#adaptive batch size
            shuffle=False, drop_last=False, num_workers=self.n_workers)

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            self.train_step(graph, blocks, input_nodes, seeds, subgraph_to_id)


    def get_model(self):
        return "random"


class PrioritizedPytorchSupervisedGraphSage(PytorchSupervisedGraphSage):
    '''
    A GraphSAGE model trained on prioritized vertices
    '''

    def __init__(self, model, batch_per_timestep, batch_size, labels, samples, priority_strategy, full_pass = 2, cuda=False, batch_full = 512, n_workers=0):
        super(PrioritizedPytorchSupervisedGraphSage, self).__init__(model, batch_per_timestep, batch_size, labels, samples, reduction='none', n_workers=n_workers, cuda=cuda, batch_full = batch_full)
        self.time_step = 0
        self.pass_var = 0
        self.full_pass = full_pass
        self.priority_strategy = priority_strategy

    def choose_vertices(self, graph_util):
        if self.time_step % (self.full_pass) == 0: #(int(self.pass_var/self.full_pass)+1) == 0:
            self.pass_var += 1
            self.recompute_priorities(graph_util, graph_util.get_train_set())
        elif len(graph_util.get_new_train_nodes()) > 1:
            #print(graph_util.get_new_train_nodes())
            self.recompute_priorities(graph_util, graph_util.get_new_train_nodes())
        batch_nodes = []
        for xx in range(self.batch_per_timestep):
            batch_nodes += graph_util.draw_priority_train_nodes(self.batch_size)
        return batch_nodes

    def _run_custom_train(self, graph, subgraph_to_id, id_to_subgraph, train_vertices, graph_util):
        '''
        Trains the prioritized graphsage for self.batch_per_timestep mini-batches
        :param graph_util:
        :return:
        '''
        #self.recompute_priorities(graph_util, graph_util.get_new_train_nodes())
        train_vertices = torch.LongTensor(train_vertices)

        sampler = dgl.sampling.MultiLayerNeighborSampler([self.samples for x in range(2)], replace=True, return_eids=True)
        dataloader = dgl.sampling.NodeDataLoader(graph, train_vertices, sampler,
                                                 batch_size=len(train_vertices) // self.batch_per_timestep,
                                                 # adaptive batch size
                                                 shuffle=False, drop_last=False, num_workers=self.n_workers)

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            batch_nodes_seed = subgraph_to_id[utils.from_nn_lib_to_numpy(seeds)]
            batch_inputs = graph.ndata['feat'][input_nodes]  # .to(device)
            batch_labels = graph.ndata['target'][seeds]  # self.labels[subgraph_to_id[seeds]]  # .to(device)

            if 'feat' in graph.edata:
                self.copy_edges_feats(graph, blocks)

            if self._cuda_var:
                batch_inputs = batch_inputs.cuda()
                batch_labels = batch_labels.cuda()
                blocks = [block.to(torch.device("cuda")) for block in blocks]

            self.optimizer.zero_grad()
            scores = self.graphsage_model(blocks, batch_inputs)
            #if batch_labels.size(0) > 1:
            #    unaggregated_loss = self.xent(scores, batch_labels.squeeze())
            #else:
            unaggregated_loss = self.xent(scores, batch_labels.flatten())

            loss = torch.mean(unaggregated_loss)
            loss.backward()
            self.optimizer.step()
            unaggregated_loss = unaggregated_loss.detach()
            priorities = self.priority_strategy.get_priorities(batch_nodes_seed, unaggregated_loss.data.cpu().numpy())
            losses = dict(zip(batch_nodes_seed, priorities))
            graph_util.update_priorities(losses)

        self.time_step += 1

    def recompute_priorities(self, graph_util, train_set):
        self.graphsage_model.eval()

        #train_set = graph_util.get_train_set()
        id_to_subgraph = graph_util.get_original_to_subgraph_map()
        subgraph_to_id = graph_util.get_subgraph_to_original_map()

        batch_nodes_seed = torch.LongTensor(id_to_subgraph[train_set])

        output_data = []
        batch_nids_l = []

        graph = graph_util.get_graph()

        sampler = dgl.sampling.MultiLayerNeighborSampler([self.samples for x in range(2)], replace=True, return_eids=True)
        dataloader = dgl.sampling.NodeDataLoader(graph, batch_nodes_seed, sampler,
                                                 batch_size=self.batch_full,
                                                 shuffle=False, drop_last=False, num_workers=self.n_workers)

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            #print(seeds)
            batch_nodes_seed = subgraph_to_id[utils.from_nn_lib_to_numpy(seeds)] #utils.from_nn_lib_to_numpy(subgraph_to_id[seeds])
            batch_inputs = graph.ndata['feat'][input_nodes]  # .to(device)
            batch_labels = graph.ndata['target'][seeds]  # self.labels[subgraph_to_id[seeds]]  # .to(device)

            if 'feat' in graph.edata:
                self.copy_edges_feats(graph, blocks)

            if self._cuda_var:
                batch_inputs = batch_inputs.cuda()
                batch_labels = batch_labels.cuda()
                blocks = [block.to(torch.device("cuda")) for block in blocks]

            scores = self.graphsage_model(blocks, batch_inputs)
            unaggregated_loss = self.xent(scores, batch_labels.flatten())
            # print(scores, " ", batch_labels.flatten(), " ", unaggregated_loss)

            batch_nids_l += list(batch_nodes_seed)
            output_data += [unaggregated_loss.detach().data.cpu().numpy()]

        unaggregated_loss = np.concatenate(output_data)
        #print(unaggregated_loss)
        priorities = self.priority_strategy.get_priorities(batch_nids_l, unaggregated_loss)
        losses = dict(zip(batch_nids_l, priorities))
        graph_util.update_priorities(losses)

    def get_model(self):
        return "prioritized"


class FullPytorchSupervisedGraphSage(PytorchSupervisedGraphSage):
    '''
    Standard offline GraphSAGE
    '''
    def __init__(self, model, batch_per_timestep, batch_size, labels, samples, cuda=False, batch_full=512, n_workers=0):
        super(FullPytorchSupervisedGraphSage, self).__init__(model, batch_per_timestep, batch_size, labels, samples, n_workers=n_workers, cuda=cuda, batch_full = batch_full)

    def choose_vertices(self, graph_util):
        return graph_util.get_train_set().copy()

    def _run_custom_train(self, graph, subgraph_to_id, id_to_subgraph, batch_nodes, graph_util):
        self.graphsage_model.train()

        train_set = torch.LongTensor(batch_nodes)
        print("LEN TRAIN SET: ", len(train_set))
        for epoch in range(self.batch_per_timestep):
            print("epoch: ", epoch, " of ", self.batch_per_timestep)
            idx = torch.randperm(train_set.nelement())
            train_set = train_set.view(-1)[idx].view(train_set.size())

            sampler = dgl.sampling.MultiLayerNeighborSampler([self.samples for x in range(2)], replace=True, return_eids=True)
            dataloader = dgl.sampling.NodeDataLoader(graph, train_set, sampler,
                                                     batch_size=self.batch_size,
                                                     # adaptive batch size
                                                     shuffle=False, drop_last=False, num_workers=self.n_workers)

            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                self.train_step(graph, blocks, input_nodes, seeds, subgraph_to_id)

    def get_model(self):
        return "offline"


class NoRehPytorchSupervisedGraphSage(PytorchSupervisedGraphSage):
    '''
    Trains GraphSAGE only on new vertices
    '''
    def __init__(self, model, batch_per_timestep, batch_size, labels, samples, cuda=False, batch_full=512, n_workers=0):
        super(NoRehPytorchSupervisedGraphSage, self).__init__(model, batch_per_timestep, batch_size, labels, samples, n_workers=n_workers, cuda=cuda, batch_full = batch_full)

    def choose_vertices(self, graph_util):
        return []

    def _run_custom_train(self, graph, subgraph_to_id, id_to_subgraph, batch_nodes, graph_util):

        self.graphsage_model.train()
        for b in range(self.batch_per_timestep):
            idxs = graph_util.get_new_train_nodes(self.batch_size)
            if len(idxs) < 2:
                return

            batch_nodes = id_to_subgraph[idxs]
            sampler = dgl.sampling.MultiLayerNeighborSampler([self.samples for x in range(2)], replace=True, return_eids=True)
            dataloader = dgl.sampling.NodeDataLoader(graph, batch_nodes, sampler,
                                                     batch_size=len(batch_nodes),
                                                     # adaptive batch size
                                                     shuffle=True, drop_last=False, num_workers=self.n_workers)

            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                self.train_step(graph, blocks, input_nodes, seeds, subgraph_to_id)


    def get_model(self):
        return "no_rehersal"
