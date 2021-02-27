from sklearn.metrics import f1_score
import sklearn
import utils
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

#from memory_profiler import profile

#import tensorflow as tf
"""
Supervised GraphSAGE model
"""

class SupervisedGraphSage():
    '''
    Base GraphSAGE class
    '''

    def __init__(self, graphsage_model, batch_per_timestep, batch_size, labels, samples, n_workers, cuda, batch_full):
        super(SupervisedGraphSage, self).__init__()

        self.graphsage_model = graphsage_model
        self.batch_size = batch_size
        self.batch_per_timestep = batch_per_timestep
        self.samples = samples
        self.n_workers = n_workers
        self.labels = labels
        self._cuda_var = cuda
        self.batch_full = batch_full
        self.amount_of_train = {}

    def build_optimizer(self):
        raise NotImplementedError

    def train_timestep(self, graph_util):
        raise NotImplementedError

    #@profile
    def evaluate(self, graph_util, path):
        test = np.array(graph_util.get_test_set())
        return self._evaluate_vertices(graph_util, path, test)

    #@profile
    def evaluate_next_snapshots(self, temporal_graph, delta, path, at_least=20):
        new_vertices, labelled = temporal_graph.get_added_vertices(delta)
        test = np.array(new_vertices)[labelled]
        print("testin over ", len(test))
        if len(test) < at_least:
            print("test skipped, too few vertices")
            f = open(path, "a+")
            f.write(self.get_model()+";;;\n")
            f.close()
            return
        return self._evaluate_vertices(temporal_graph, path, test)

    def _evaluate_vertices(self, graph_util, path, batch_nids):
        '''

        :param graph_util: A TrainTestGraph
        :param path: Results path
        :param vertices: vertices that should be tested
        :return:
        '''

        id_to_subgraph = graph_util.get_original_to_subgraph_map()
        subgraph_to_id = graph_util.get_subgraph_to_original_map()
        graph = graph_util.get_graph()
        #graph.readonly(readonly_state=True)
        vertices = id_to_subgraph[batch_nids]#[id_to_subgraph[v] for v in batch_nids]
        #with tf.device('/GPU:0'):
        print("run custom eval start")
        output_data = self._run_custom_eval(graph, subgraph_to_id, id_to_subgraph, vertices)
        print("end")
        output_data = np.concatenate(output_data)

        if len(output_data) == 0:
            #graph.readonly(readonly_state=False)
            return

        labels = utils.from_nn_lib_to_numpy(utils.index_tensor(graph.ndata['target'], vertices))
        confusion_matrix = sklearn.metrics.confusion_matrix(labels, output_data.argmax(axis=1))
        confusion_matrix = [item for sublist in confusion_matrix for item in sublist]
        f1 = f1_score(labels, output_data.argmax(axis=1), average="macro")

        f = open(path, "a+")
        f.write(self.get_model()+";"+str(f1)+";"+str(self.delay)+";"+str(confusion_matrix)+"\n")
        f.close()

        print("Test F1:", f1)
        #graph.readonly(readonly_state=False)
        #print("done")


    def _run_custom_eval(self):
        raise NotImplementedError

    def get_model(self):
        return "base_model"

    def choose_vertices(self, graph_util):
        raise NotImplementedError

    #@profile
    def train_timestep(self, graph_util):
        batch_nodes = self.choose_vertices(graph_util)
        start = time.time()
        id_to_subgraph = graph_util.get_original_to_subgraph_map()
        subgraph_to_id = graph_util.get_subgraph_to_original_map()
        graph = graph_util.get_graph()
        #graph.readonly(readonly_state=True)
        #with tf.device('/GPU:0'):
        self._run_custom_train(graph, subgraph_to_id, id_to_subgraph, id_to_subgraph[batch_nodes], graph_util)
        self.delay = time.time() - start
        #graph.readonly(readonly_state=False)

    #@profile
    def generate_tsne(self, graph_util, folder, index):
        '''
        Generates the TSNE plot
        :param graph_util: A TrainTestGraph
        :param folder: folder where the plot should be stored
        :param index: snapshot value (or another suffix)
        :return:
         '''
        train_set = graph_util.get_train_set()

        id_to_subgraph = graph_util.get_original_to_subgraph_map()
        subgraph_to_id = graph_util.get_subgraph_to_original_map()

        graph = graph_util.get_graph()
        #graph.readonly(readonly_state=True)

        vertices = id_to_subgraph[train_set]
        targets_plot = utils.from_nn_lib_to_numpy(utils.index_tensor(graph.ndata['target'],vertices)) #(self.labels[train_set])

        output_data = self._run_custom_eval(graph, subgraph_to_id, id_to_subgraph, vertices)

        #train_set = batch_nodes_seed_list

        #graph.readonly(readonly_state=False)

        emb = np.concatenate(output_data)  # nodes embeddings
        y = targets_plot #emb.argmax(axis=1)

        trans_ret = TSNE(n_components=2, random_state=1)
        emb_transformed = pd.DataFrame(trans_ret.fit_transform(emb), index=list(range(len(targets_plot))))
        emb_transformed['class'] = y
        emb_transformed['class'] = 'class ' + emb_transformed['class'].astype(str)
        #print(emb_transformed['class'])

        emb_transformed['priority'] = graph_util.dump_priorities(train_set)

        amount_train = []
        # added code
        for v in train_set:
            if v in self.amount_of_train:
                amount_train += [self.amount_of_train[v]]
            else:
                amount_train += [0]

        emb_transformed['train_passes'] = amount_train
        # end

        emb_transformed = emb_transformed.rename(columns={0: "x0", 1: "x1"})
        emb_transformed = emb_transformed.sort_values(by=['priority'], ascending=True)

        emb_transformed.to_csv(folder + "/data_" + str(index) + ".csv", index=False)

        sns.set_style("ticks")
        plt.figure()
        pl = sns.scatterplot(x="x0", y="x1", hue="class", data=emb_transformed, linewidth=0.5, s=40)  # linewidth=0)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig = pl.get_figure()
        fig.savefig(folder + "/tsne_class_" + str(index) + ".pdf")
        plt.figure()
        # hue_norm=LogNorm()
        p2 = sns.scatterplot(x="x0", y="x1", hue="priority", alpha=0.2, data=emb_transformed,
                             linewidth=0.1, s=40)  # linewidth=0)
        fig2 = p2.get_figure()
        fig2.savefig(folder + "/tsne_priority_" + str(index) + ".pdf")

        # ----
        '''
        plt.figure()
        p2 = sns.scatterplot(x="x0", y="x1", hue="train_passes", alpha=0.2, data=emb_transformed,
                             linewidth=0.1, s=40)  # linewidth=0)
        fig2 = p2.get_figure()
        fig2.savefig(folder + "/tsne_train_passes_" + str(index) + ".pdf")
        
        self.amount_of_train = {}
        '''

        plt.close('all')
