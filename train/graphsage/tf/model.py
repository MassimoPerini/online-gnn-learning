import numpy as np
import dgl
import tensorflow as tf
from graphsage.model import SupervisedGraphSage
import utils
import torch

"""
Supervised GraphSAGE model
"""

class TFSupervisedGraphSage(SupervisedGraphSage):
    '''
    Several GraphSAGE train methods
    '''

    def __init__(self, graphsage_model, batch_per_timestep, batch_size, labels, samples, reduction=tf.keras.losses.Reduction.AUTO, n_workers = 1, cuda=False, batch_full = 512):
        super(TFSupervisedGraphSage, self).__init__(graphsage_model, batch_per_timestep, batch_size, labels, samples, n_workers, cuda, batch_full)
        self.xent = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=reduction)

    def build_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)


    def _run_custom_eval(self, graph, subgraph_to_id, id_to_subgraph, test_vertices):
        output_data = []
        batch_nodes_seed = utils.to_nn_lib(test_vertices, GPU=False, dtype=tf.int64)

        sampler = dgl.sampling.MultiLayerNeighborSampler([self.samples for x in range(2)], replace=True)

        collator = dgl.sampling.NodeCollator(graph, batch_nodes_seed, sampler)
        dataloader = torch.utils.data.DataLoader(collator.dataset, collate_fn=collator.collate,
                                                 batch_size=self.batch_full,
                                                 shuffle=False, drop_last=False, num_workers=self.n_workers)

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            batch_inputs = utils.index_tensor(graph.ndata['feat'], input_nodes)  # .to(device)
            val_output = self.graphsage_model(blocks, batch_inputs)
            output_data += [val_output.numpy()]

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

        batch_inputs = utils.index_tensor(graph.ndata['feat'], input_nodes)
        batch_labels = utils.index_tensor(graph.ndata['target'], seeds)

        with tf.GradientTape() as tape:
            scores = self.graphsage_model(blocks, batch_inputs, training=True)
            loss = self.xent(tf.squeeze(batch_labels), scores)
        gradients = tape.gradient(loss, self.graphsage_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.graphsage_model.trainable_variables))


class RandomTfSupervisedGraphSage(TFSupervisedGraphSage):
    '''
    A GraphSAGE model trained on random vertices
    '''

    def __init__(self, model, batch_per_timestep, batch_size, labels, samples, cuda=False, batch_full=512, n_workers=3):
        super(RandomTfSupervisedGraphSage, self).__init__(model, batch_per_timestep, batch_size, labels, samples, n_workers=n_workers, cuda=cuda, batch_full = batch_full)

    def choose_vertices(self, graph_util):
        batch_nodes = []
        for xx in range(self.batch_per_timestep):
            batch_nodes += graph_util.draw_random_train_nodes(self.batch_size)
        return batch_nodes

    def _run_custom_train(self, graph, subgraph_to_id, id_to_subgraph, train_vertices, graph_util):

        train_vertices = utils.to_nn_lib(train_vertices, GPU=False, dtype=tf.int64)

        sampler = dgl.sampling.MultiLayerNeighborSampler([self.samples for x in range(2)], replace=True)
        collator = dgl.sampling.NodeCollator(graph, train_vertices, sampler)
        dataloader = torch.utils.data.DataLoader(collator.dataset, collate_fn = collator.collate,batch_size=len(train_vertices)//self.batch_per_timestep,#adaptive batch size
            shuffle=False, drop_last=False, num_workers=self.n_workers)
        ### TF Data is executed in "graph mode", not Eager Execution (despite the fact that EE should be the default exec. mode)
        #Graph mode is not supported -> use pytorch loader

        #dataloader = tf.data.Dataset.from_tensors(collator.dataset)
        #dataloader = dataloader.map(collator.collate, num_parallel_calls=(None if self.n_workers == 0 else self.n_workers), deterministic=True)#self.n_workers
        #dataloader = dataloader.batch(len(train_vertices)//self.batch_per_timestep, drop_remainder=False)
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            self.train_step(graph, blocks, input_nodes, seeds, subgraph_to_id)

    def get_model(self):
        return "random"


class PrioritizedTfSupervisedGraphSage(TFSupervisedGraphSage):
    '''
    A GraphSAGE model trained on prioritized vertices
    '''

    def __init__(self, model, batch_per_timestep, batch_size, labels, samples, priority_strategy, full_pass = 2, cuda=False, batch_full = 512, n_workers=3):
        super(PrioritizedTfSupervisedGraphSage, self).__init__(model, batch_per_timestep, batch_size, labels, samples, reduction='none', n_workers=n_workers, cuda=cuda, batch_full = batch_full)
        self.time_step = 0
        self.pass_var = 0
        self.full_pass = full_pass
        self.priority_strategy = priority_strategy

    def choose_vertices(self, graph_util):
        if self.time_step % self.full_pass == 0:  # (int(self.pass_var/self.full_pass)+1) == 0:
            self.pass_var += 1
            self.recompute_priorities(graph_util)
        batch_nodes = []
        for xx in range(self.batch_per_timestep):
            batch_nodes += graph_util.draw_priority_train_nodes(self.batch_size)
        return batch_nodes

    def _run_custom_train(self, graph, subgraph_to_id, id_to_subgraph, batch_nodes, graph_util):
        train_vertices = utils.to_nn_lib(batch_nodes, GPU=False, dtype=tf.int64)

        sampler = dgl.sampling.MultiLayerNeighborSampler([self.samples for x in range(2)], replace=True)
        collator = dgl.sampling.NodeCollator(graph, train_vertices, sampler)
        dataloader = torch.utils.data.DataLoader(collator.dataset, collate_fn = collator.collate, batch_size=len(train_vertices) // self.batch_per_timestep,#adaptive batch size
            shuffle=False, drop_last=False, num_workers=self.n_workers)

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            batch_nodes_seed = utils.from_nn_lib_to_numpy(subgraph_to_id[seeds])
            batch_inputs = utils.index_tensor(graph.ndata['feat'], input_nodes)
            batch_labels = utils.index_tensor(graph.ndata['target'], seeds)

            with tf.GradientTape() as tape:
                scores = self.graphsage_model(blocks, batch_inputs, training=True)
                unaggregated_loss = self.xent(tf.squeeze(batch_labels), scores)
                loss = tf.math.reduce_mean(unaggregated_loss)

            gradients = tape.gradient(loss, self.graphsage_model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.graphsage_model.trainable_variables))
            priorities = self.priority_strategy.get_priorities(batch_nodes_seed, unaggregated_loss.numpy())
            losses = dict(zip(batch_nodes_seed, priorities))
            graph_util.update_priorities(losses)

        self.time_step += 1


    def recompute_priorities(self, graph_util):
        train_set = graph_util.get_train_set()

        id_to_subgraph = graph_util.get_original_to_subgraph_map()
        subgraph_to_id = graph_util.get_subgraph_to_original_map()
        batch_nodes_seed_ = utils.to_nn_lib(id_to_subgraph[train_set], GPU=False, dtype=tf.int64)

        output_data = []
        batch_nids_l = []
        graph = graph_util.get_graph()

        sampler = dgl.sampling.MultiLayerNeighborSampler([self.samples for x in range(2)], replace=True)
        collator = dgl.sampling.NodeCollator(graph, batch_nodes_seed_, sampler)
        dataloader = torch.utils.data.DataLoader(collator.dataset, collate_fn = collator.collate, batch_size=self.batch_full,#adaptive batch size
            shuffle=False, drop_last=False, num_workers=self.n_workers)

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            batch_nodes_seed = utils.from_nn_lib_to_numpy(subgraph_to_id[seeds])
            batch_inputs = utils.index_tensor(graph.ndata['feat'],input_nodes)  # .to(device)
            batch_labels = utils.index_tensor(graph.ndata['target'], seeds)  # self.labels[subgraph_to_id[seeds]]  # .to(device)

            scores = self.graphsage_model(blocks, batch_inputs)
            unaggregated_loss = self.xent(tf.squeeze(batch_labels), scores)

            batch_nids_l += list(batch_nodes_seed)
            output_data += [unaggregated_loss.numpy()]

        unaggregated_loss = np.concatenate(output_data)
        priorities = self.priority_strategy.get_priorities(batch_nids_l, unaggregated_loss)
        losses = dict(zip(batch_nids_l, priorities))
        graph_util.update_priorities(losses)

    def get_model(self):
        return "prioritized"


class FullTfSupervisedGraphSage(TFSupervisedGraphSage):
    '''
    Standard offline GraphSAGE
    '''
    def __init__(self, model, batch_per_timestep, batch_size, labels, samples, cuda=False, batch_full=512, n_workers=3):
        super(FullTfSupervisedGraphSage, self).__init__(model, batch_per_timestep, batch_size, labels, samples, n_workers=n_workers, cuda=cuda, batch_full = batch_full)

    def choose_vertices(self, graph_util):
        return graph_util.get_train_set().copy()

    def _run_custom_train(self, graph, subgraph_to_id, id_to_subgraph, batch_nodes, graph_util):

        train_set = utils.to_nn_lib( batch_nodes, GPU=False, dtype=tf.int64)
        print("LEN TRAIN SET: ", len(train_set))
        for epoch in range(self.batch_per_timestep):
            print("epoch: ", epoch, " of ", self.batch_per_timestep)
            train_set = tf.random.shuffle(train_set)

            sampler = dgl.sampling.MultiLayerNeighborSampler([self.samples for x in range(2)], replace=True)

            collator = dgl.sampling.NodeCollator(graph, train_set, sampler)
            dataloader = torch.utils.data.DataLoader(collator.dataset, collate_fn=collator.collate,
                                                     batch_size=self.batch_size,  # adaptive batch size
                                                     shuffle=False, drop_last=False, num_workers=self.n_workers)

            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                self.train_step(graph, blocks, input_nodes, seeds, subgraph_to_id)


    def get_model(self):
        return "offline"


class NoRehTfSupervisedGraphSage(TFSupervisedGraphSage):
    '''
    Trains GraphSAGE only on new vertices
    '''
    def __init__(self, model, batch_per_timestep, batch_size, labels, samples, cuda=False, batch_full=512, n_workers=3):
        super(NoRehTfSupervisedGraphSage, self).__init__(model, batch_per_timestep, batch_size, labels, samples, n_workers=n_workers, cuda=cuda, batch_full = batch_full)

    def choose_vertices(self, graph_util):
        return []

    def _run_custom_train(self, graph, subgraph_to_id, id_to_subgraph, batch_nodes, graph_util):

        for b in range(self.batch_per_timestep):
            batch_nodes = id_to_subgraph[graph_util.get_new_train_nodes(self.batch_size)]
            if len(batch_nodes) < 2:
                return

            sampler = dgl.sampling.MultiLayerNeighborSampler([self.samples for x in range(2)], replace=True)

            collator = dgl.sampling.NodeCollator(graph, batch_nodes, sampler)
            dataloader = torch.utils.data.DataLoader(collator.dataset, collate_fn=collator.collate,
                                                     batch_size=len(batch_nodes),  # adaptive batch size
                                                     shuffle=True, drop_last=False, num_workers=self.n_workers)

            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                self.train_step(graph, blocks, input_nodes, seeds, subgraph_to_id)

    def get_model(self):
        return "no_rehersal"
