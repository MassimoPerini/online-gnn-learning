'''
Represents a dynamic graph. Supports vertex timestamps
'''

from graph.dynamic_graph import DynamicGraph
import dgl
import numpy as np
import utils

class DynamicGraphEdge(DynamicGraph):
    '''
        An edge addition graph (timestamps on the vertices)
    '''

    def __init__(self, snapshots, labelled_vertices, search_depth=1):
        '''
        :param graph: DGL graph
        :param snapshots: snapshots: n. snapshots (>=1)
        :param labels: labels of the vertices. -1 == NO LABEL (the vertex won't be trained)
        :param search_depth: how many <search_depth>-hop neighbours are affected by an update
        '''

        super(DynamicGraphEdge, self).__init__(None, snapshots, labelled_vertices, search_depth)
        self.snapshot_edges = None
        self.new_vertices = set() #vertices added in the last round
        self.evolving_vertices = set() #keep all vertices added
        self.current_subgraph = dgl.graph([])
        self.edge_feats = None

    def build(self, vertex_feats, targets, cuda, edge_timestamps=None, ensure_labelled=None, restrict=None, edge_feats = None):
        '''

        :param edge_timestamps: edge timestamp as a map (key=(first vertex, second vertex), value=timestamp)
        :param ensure_labelled: ensure at least a given amount of new labelled vertices per snapshot (values: None, 0-1) (not implemented)
        :param restrict: ignore edges with timestamp greater than the X-th edge
        :return:
        '''
        self.vertex_feats = vertex_feats
        self.targets = targets

        if edge_timestamps is None:
            self._generate_snapshot_random()
        else:# isinstance(edge_timestamps, pd.DataFrame):
            self._generate_snapshot_pandas(edge_timestamps, edge_feats, cuda, ensure_labelled, restrict=restrict)

    def _generate_snapshot_pandas(self, edge_timestamps, edge_feats, cuda, ensure_labelled=None, restrict = None):
        src = edge_timestamps["src"].values
        dst = edge_timestamps["dst"].values

        if restrict is not None and restrict < len(edge_timestamps):
            src = src[:restrict]
            dst = dst[:restrict]
            if edge_feats is not None:
                edge_feats = edge_feats[:restrict]

        self.src = src
        self.dst = dst
        self.edge_feats = edge_feats
        self.edges_per_snapshot = int(len(edge_timestamps) / self.snapshots)
        print("edges per snapshot: ", self.edges_per_snapshot)
        new_vertices = np.sort(np.unique(np.concatenate([self.src[0:self.edges_per_snapshot], self.dst[0:self.edges_per_snapshot]])))
        print("new vertices", new_vertices)
        print(len(new_vertices))
        self.current_subgraph.add_nodes(len(new_vertices), {'feat': utils.to_nn_lib(utils.index_tensor(self.vertex_feats, new_vertices), cuda),
                                        'target': utils.to_nn_lib(utils.index_tensor(self.targets, new_vertices), cuda)})
        print("add nodes")
        if self.edge_feats is not None:
            self.current_subgraph.add_edges(self.src[0:self.edges_per_snapshot], self.dst[0:self.edges_per_snapshot], {'feat': utils.to_nn_lib(self.edge_feats[0:self.edges_per_snapshot], cuda)})
            self.current_subgraph.add_edges(self.dst[0:self.edges_per_snapshot], self.src[0:self.edges_per_snapshot], {'feat': utils.to_nn_lib(self.edge_feats[0:self.edges_per_snapshot], cuda)})
        else:
            self.current_subgraph.add_edges(self.src[0:self.edges_per_snapshot], self.dst[0:self.edges_per_snapshot])
            self.current_subgraph.add_edges(self.dst[0:self.edges_per_snapshot], self.src[0:self.edges_per_snapshot])

        print("to nn")
        self.evolving_vertices = utils.from_nn_lib_to_set(self.current_subgraph.nodes())  #
        self.new_vertices = self.evolving_vertices.copy()
        self.evolution_index = 1
        self.subgraph_to_original_map = Wrap()
        self.original_to_subgraph_map = self.subgraph_to_original_map
        #if cuda:
        #    self.current_subgraph = self.current_subgraph.to(utils.get_context())
        print("builded, size: ", len(self.evolving_vertices))

    def _generate_snapshot_random(self):
        raise NotImplementedError

    def get_added_vertices(self, delta=None):
        '''
        :return: The NEW vertices
        '''

        if self.snapshot_edges is None:
            return self.get_added_vertices_pandas(delta)
        if delta is None:
            vertices = self.new_vertices

        else:
            vertices = set()
            for i in range(delta):
                for edge in self.snapshot_edges[self.evolution_index-i-1]:
                    v1 = edge[0]
                    v2 = edge[1]
                    vertices.add(v1)
                    vertices.add(v2)
            vertices = list(vertices)

        labelled = []
        for v in vertices:
            labelled += [v in self.labelled_vertices]
        return vertices, labelled

    def get_added_vertices_pandas(self, delta=None):
        if delta is None:
            vertices = self.new_vertices

        else:
            vertices1 = self.src[(self.evolution_index - delta) * self.edges_per_snapshot: (
                                                                                               self.evolution_index) * self.edges_per_snapshot]
            vertices2 = self.dst[(self.evolution_index - delta) * self.edges_per_snapshot: (
                                                                                               self.evolution_index) * self.edges_per_snapshot]
            vertices = np.unique(np.concatenate([vertices1, vertices2]))

        labelled = [v in self.labelled_vertices for v in vertices]
        return vertices, labelled



    def get_graph(self):
        return self.current_subgraph
        #return self.sub_g

    def __len__(self):
        return self.snapshots #len(self.snapshot_edges)

    '''
    def evolve(self):

        if self.snapshot_edges is None:
            return self.evolve_dataframe()

        unzipped = list(zip(*self.snapshot_edges[self.evolution_index]))

        new_vertices = set(unzipped[0])
        new_vertices.update(unzipped[1])
        new_vertices = list(new_vertices)
        new_vertices.sort()

        self.new_vertices = set()
        sorted_new_vertices = []

        for edge in self.snapshot_edges[self.evolution_index]:
            v1 = edge[0]
            v2 = edge[1]
            if v1 not in self.evolving_vertices and v1 not in self.new_vertices:
                self.new_vertices.add(v1)
                sorted_new_vertices += [v1]
            if v2 not in self.evolving_vertices and v2 not in self.new_vertices:
                self.new_vertices.add(v2)
                sorted_new_vertices += [v2]

            self.evolving_vertices.add(v1)
            self.evolving_vertices.add(v2)


        for xx in range(1, len(sorted_new_vertices)):
            assert(sorted_new_vertices[xx] > sorted_new_vertices[xx-1])
        #self.current_subgraph.readonly(False)
        #self.current_subgraph.add_nodes(len(self.new_vertices), {'feat': self.graph.nodes[sorted_new_vertices].data['feat']})
        self.current_subgraph.add_nodes(len(self.new_vertices),
                                        {'feat': self.vertex_feats[sorted_new_vertices]})
        if self.edge_feats is None:
            self.current_subgraph.add_edges(unzipped[0], unzipped[1])#(self.snapshot_edges[self.evolution_index])
            self.current_subgraph.add_edges(unzipped[1], unzipped[0])#TODO undirected only
        else:
            start = int(self.current_subgraph.number_of_edges()/2)
            self.current_subgraph.add_edges(unzipped[0], unzipped[1], {'feat': self.edge_feats[start:start+len(unzipped[0])]})
            self.current_subgraph.add_edges(unzipped[1], unzipped[0], {'feat': self.edge_feats[start:start+len(unzipped[0])]})
        #if self.features_vertex is not None:
        l_new_vertices = list(self.new_vertices)

        #g_dgl.nodes[l_sorted_vertices[i:i+50000]].data

        #self.current_subgraph.nodes[l_new_vertices].data['feat'] = self.graph.nodes[l_new_vertices].data['feat']
        self.current_subgraph.nodes[l_new_vertices].data['feat'] = self.vertex_feats[l_new_vertices]

        self.evolution_index += 1
        #self.current_subgraph.readonly(True)
    '''

    def evolve(self):
        #print("edges/snapshot: ", self.edges_per_snapshot)
        print(self.evolution_index * self.edges_per_snapshot, " : ", (self.evolution_index + 1)* self.edges_per_snapshot)
        new_vertices1 = self.src[self.evolution_index * self.edges_per_snapshot :(self.evolution_index + 1)* self.edges_per_snapshot]
        new_vertices2 = self.dst[self.evolution_index * self.edges_per_snapshot:(self.evolution_index + 1)* self.edges_per_snapshot]
        new_vertices = np.sort(np.unique(np.concatenate([new_vertices1, new_vertices2])))

        self.new_vertices = set()
        sorted_new_vertices = []

        for x in new_vertices:
            if x not in self.evolving_vertices and x not in self.new_vertices:
                self.new_vertices.add(x)
                sorted_new_vertices += [x]
            self.evolving_vertices.add(x)
        #self.current_subgraph.readonly(False)
        self.current_subgraph.add_nodes(len(self.new_vertices), {'feat': utils.index_tensor(self.vertex_feats, sorted_new_vertices),
                                                                 'target': utils.index_tensor(self.targets, sorted_new_vertices)})
        if self.edge_feats is not None:
            self.current_subgraph.add_edges(new_vertices1, new_vertices2,
                                            {'feat': self.edge_feats[self.evolution_index * self.edges_per_snapshot : (self.evolution_index + 1)* self.edges_per_snapshot]})
            self.current_subgraph.add_edges(new_vertices2, new_vertices1,
                                            {'feat': self.edge_feats[self.evolution_index * self.edges_per_snapshot :(self.evolution_index + 1)* self.edges_per_snapshot]})
        else:
            self.current_subgraph.add_edges(new_vertices1, new_vertices2)
            self.current_subgraph.add_edges(new_vertices2, new_vertices1)
        l_new_vertices = list(self.new_vertices)
        #self.current_subgraph.nodes[l_new_vertices].data['feat'] = utils.index_tensor(self.vertex_feats, l_new_vertices)
        self.evolution_index += 1


    def get_original_to_subgraph_map(self):
        '''
        This class uses DGL subgraphs to generate the temporal graph (the graph at a given timestep is a subgraph of the
        complete graph). However DGL updates the vertex ids of the subgraph, therefore they do not match with the ids of the original graph.
        This dictionary maps the original vertex ids to the subgraph vertex id
        :return:
        '''
        return self.original_to_subgraph_map

    def get_subgraph_to_original_map(self):
        '''
        This class uses DGL subgraphs to generate the temporal graph (the graph at a given timestep is a subgraph of the
        complete graph). However DGL updates the vertex ids of the subgraph, therefore they do not match with the ids of the original graph.
        This dictionary maps the subgraph vertex id to the original graph vertex id
        :return:
        '''
        return self.subgraph_to_original_map

    def get_vertices_changed(self):
        '''

        :return: Every vertex affected by changes (new vertices + vertices affected by new edges)
        '''

        if self.snapshot_edges is None:
            return self.get_vertices_changed_dataframe()

        vertices = set()
        edgelist = self.snapshot_edges[self.evolution_index - 1]
        for edge in edgelist:
            vertices.add(edge[0])
            vertices.add(edge[1])

        return vertices, self.search_depth

    def get_vertices_changed_dataframe(self):
        new_vertices1 = self.src[(self.evolution_index - 1) * self.edges_per_snapshot:
                                                                                            self.evolution_index * self.edges_per_snapshot]
        new_vertices2 = self.dst[(self.evolution_index -1) * self.edges_per_snapshot:
                                                                                            self.evolution_index * self.edges_per_snapshot]
        return set(np.unique(np.concatenate([new_vertices1, new_vertices2]))), self.search_depth

class Wrap():
    def __getitem__(self, item):
        return item



