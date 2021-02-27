import numpy as np
import random
import dgl
import tensorflow as tf
from graphsage.model import SupervisedGraphSage
import utils
import torch

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
            h, adj_matrix, real_shapes = self.pad_features(batch_inputs, blocks, batch_size=self.batch_full)
            real_d = tf.tuple([tf.constant(x) for x in real_shapes])

            val_output = self._test_step(h, adj_matrix, real_d)
            output_data += [val_output.numpy()]

        return output_data

    @tf.function(experimental_relax_shapes=True)
    def _test_step(self, tensors, matrices, real_batch):
        tensors, matrices = self._get_real_dims(tensors, matrices, real_batch)
        return self.graphsage_model(tensors, matrices, real_batch)

    def get_model(self):
        return "base_model"

    def pad_features(self, batch_inputs, blocks, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        sizes = [batch_size]
        for x in range(self.graphsage_model.get_n_layers()):
            sizes += [sizes[-1]*self.samples]
        sizes.reverse()

        h = []

        real_shapes = []

        nodes_feats = batch_inputs[:blocks[0].number_of_src_nodes()]
        h += [tf.pad(nodes_feats, tf.constant([[0, sizes[0] - nodes_feats.shape[0]], [0, 0]]))]
        real_shapes += [nodes_feats.shape[0]]

        for j in range(0, self.graphsage_model.get_n_layers()):
            nodes_feats = batch_inputs[:blocks[j].number_of_dst_nodes()]
            real_shapes += [blocks[j].number_of_dst_nodes()]
            h += [tf.pad(nodes_feats, tf.constant([[0, sizes[j+1] - nodes_feats.shape[0]], [0, 0]]))]

        adj_matrix = []
        j = 0
        for block in blocks:
            matrix = block.adjacency_matrix()
            adj_matrix += [tf.sparse.SparseTensor(matrix.indices, matrix.values, [sizes[j + 1], sizes[j]])]
            j += 1

        return h, adj_matrix, real_shapes

    def train_step(self, graph, blocks, input_nodes, seeds, subgraph_to_id):
        batch_inputs = utils.index_tensor(graph.ndata['feat'], input_nodes)
        batch_labels = utils.index_tensor(graph.ndata['target'], seeds)
        h, adj_matrix, real_shapes = self.pad_features(batch_inputs, blocks)
        batch_labels = tf.pad(batch_labels, tf.constant([[0, self.batch_size - batch_labels.shape[0]], [0, 0]]))
        real_d = tf.tuple([tf.constant(x) for x in real_shapes])
        self._train_step(tf.tuple(h), tf.tuple(adj_matrix), batch_labels, real_d)

    def _get_real_dims(self, tensors, matrices, real_batch, batch_labels=None):
        tensors = list(tensors)
        matrices = list(matrices)

        for x in range(len(real_batch) - 1):
            tensors[x] = tensors[x][:real_batch[x]]
            matrices[x] = tf.sparse.slice(matrices[x], [0, 0], [real_batch[x + 1], real_batch[x]])
        tensors[-1] = tensors[-1][:real_batch[-1]]
        if batch_labels is not None:
            batch_labels = batch_labels[:real_batch[-1]]
            return tensors, matrices, batch_labels
        return tensors, matrices

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, tensors, matrices, batch_labels, real_batch):
        tensors, matrices, batch_labels = self._get_real_dims(tensors, matrices, real_batch, batch_labels)
        with tf.GradientTape() as tape:
            scores = self.graphsage_model(tensors, matrices, real_batch, training=True)
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
        dataloader = torch.utils.data.DataLoader(collator.dataset, collate_fn=collator.collate,
                                                 batch_size=len(train_vertices) // self.batch_per_timestep,
                                                 shuffle=False, drop_last=False, num_workers=self.n_workers)

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
        batch_nodes_seed = utils.to_nn_lib(batch_nodes, GPU=False, dtype=tf.int64)

        sampler = dgl.sampling.MultiLayerNeighborSampler([self.samples for x in range(2)], replace=True)
        collator = dgl.sampling.NodeCollator(graph, batch_nodes_seed, sampler)
        dataloader = torch.utils.data.DataLoader(collator.dataset, collate_fn=collator.collate,
                                                 batch_size=len(batch_nodes_seed) // self.batch_per_timestep,
                                                 shuffle=False, drop_last=False, num_workers=self.n_workers)

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            batch_nodes_seed = utils.from_nn_lib_to_numpy(subgraph_to_id[seeds])
            batch_inputs = utils.index_tensor(graph.ndata['feat'], input_nodes)
            batch_labels = utils.index_tensor(graph.ndata['target'], seeds)

            h, adj_matrix, real_shapes = self.pad_features(batch_inputs, blocks)
            batch_labels = tf.pad(batch_labels, tf.constant([[0, self.batch_size - batch_labels.shape[0]], [0, 0]]))
            real_d = tf.tuple([tf.constant(x) for x in real_shapes])

            unaggregated_loss = self._train_step(tf.tuple(h), tf.tuple(adj_matrix), batch_labels, real_d)

            priorities = self.priority_strategy.get_priorities(batch_nodes_seed, unaggregated_loss.numpy())
            losses = dict(zip(batch_nodes_seed, priorities))
            graph_util.update_priorities(losses)

        self.time_step += 1


    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, tensors, matrices, batch_labels, real_batch):
        tensors, matrices, batch_labels = self._get_real_dims(tensors, matrices, real_batch, batch_labels)
        with tf.GradientTape() as tape:
            scores = self.graphsage_model(tensors, matrices, real_batch, training=True)
            unaggregated_loss = self.xent(tf.squeeze(batch_labels), scores)
            loss = tf.math.reduce_mean(unaggregated_loss)

        gradients = tape.gradient(loss, self.graphsage_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.graphsage_model.trainable_variables))
        return unaggregated_loss

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
        dataloader = torch.utils.data.DataLoader(collator.dataset, collate_fn=collator.collate,
                                                 batch_size=self.batch_full,  # adaptive batch size
                                                 shuffle=False, drop_last=False, num_workers=self.n_workers)

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            batch_nodes_seed = utils.from_nn_lib_to_numpy(subgraph_to_id[seeds])
            batch_inputs = utils.index_tensor(graph.ndata['feat'],input_nodes)  # .to(device)
            batch_labels = utils.index_tensor(graph.ndata['target'], seeds)  # self.labels[subgraph_to_id[seeds]]  # .to(device)

            if len(batch_nodes_seed) <= 1:
                print("breaking recompute prior")
                break

            h, adj_matrix, real_shapes = self.pad_features(batch_inputs, blocks, batch_size=self.batch_full)
            batch_labels = tf.pad(batch_labels, tf.constant([[0, self.batch_full - batch_labels.shape[0]], [0, 0]]))
            real_d = tf.tuple([tf.constant(x) for x in real_shapes])

            unaggregated_loss = self._eval(h, adj_matrix, batch_labels, real_d)
            batch_nids_l += list(batch_nodes_seed)
            output_data += [unaggregated_loss.numpy()]

        unaggregated_loss = np.concatenate(output_data)

        priorities = self.priority_strategy.get_priorities(batch_nids_l, unaggregated_loss)

        losses = dict(zip(batch_nids_l, priorities))
        graph_util.update_priorities(losses)

    @tf.function
    def _eval(self, tensors, matrices, batch_labels, real_batch):
        tensors, matrices, batch_labels = self._get_real_dims(tensors, matrices, real_batch, batch_labels)
        scores = self.graphsage_model(tensors, matrices, real_batch)
        unaggregated_loss = self.xent(tf.squeeze(batch_labels), scores)
        return unaggregated_loss


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
                                                     batch_size=self.batch_size,
                                                     shuffle=False, drop_last=False, num_workers=self.n_workers)

            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                if len(seeds) % self.batch_size != 0:
                    print("breaking...")
                    break

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
