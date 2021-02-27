'''
Represents a dynamic graph. Supports vertex timestamps
'''

from graph.dynamic_graph import DynamicGraph
import dgl
import utils
import numpy
import scipy

class DynamicGraphVertex(DynamicGraph):
    '''
    A vertex addition graph (timestamps on the vertices)
    '''

    def __init__(self, graph, snapshots, labelled_vertices, search_depth = 2):
        '''
        :param graph: DGL graph
        :param snapshots: snapshots: n. snapshots (>=1)
        :param labels: labels of the vertices. -1 == NO LABEL (the vertex won't be trained)
        :param search_depth: how many <search_depth>-hop neighbours are affected by an update
        '''

        super(DynamicGraphVertex, self).__init__(graph, snapshots, labelled_vertices, search_depth)

        self.evolving_vertices = None

        print("graph size: ", len(graph), "vertices, snapshots: ", snapshots)
        self.vertex_per_snapshot = int(len(self.graph) / self.snapshots)
        print("vertex per snapshot: ", self.vertex_per_snapshot)

    def build(self, vertex_timestamps=None, ensure_labelled=None):
        if vertex_timestamps is None:
            self._generate_snapshot_random()
        else:
            self._generate_snapshot(vertex_timestamps, ensure_labelled)
        #self.sub_g.readonly(True)

    def _generate_snapshot(self, vertex_timestamps, ensure_labelled=None):
        '''
        Init the graph applying the first timestamp (vertex_timestamps)
        :param vertex_timestamps: vertex timestamp as a map (key=vertex id, value=timestamp)
        ;param ensure_labelled: ensure at least a given amount of new labelled vertices per snapshot (values: None, 0-1)
        :return:
        '''

        def sort_second(val):
            return val[1]

        ordered_vertices = list(vertex_timestamps.items())
        ordered_vertices.sort(key=sort_second)

        vertices = [x[0] for x in ordered_vertices] #keep sorted vertex ids

        if ensure_labelled is None:
            self.snapshot_vertices = [(vertices[i:i + self.vertex_per_snapshot]) for i in
                             range(0, len(self.graph), self.vertex_per_snapshot)]  # generate vertex snapshots (sublists)

        else:
            #ensures self.vertex_per_snapshot * ensure_labelled labelled vertices per partition
            assert (ensure_labelled >= 0 and ensure_labelled <= 1)

            labelled_vertices_per_snapshot = int(self.vertex_per_snapshot * ensure_labelled)

            self.snapshot_vertices = [[]]
            l_vertices_counter = 0

            for v in vertices:
                if v in self.labelled_vertices:
                    l_vertices_counter += 1

                self.snapshot_vertices[-1] += [v]

                if l_vertices_counter == labelled_vertices_per_snapshot:
                    l_vertices_counter = 0
                    self.snapshot_vertices += [[]]
            if len(self.snapshot_vertices[-1]) == 0:
                self.snapshot_vertices.pop(-1)

        # apply first snapshot

        self.evolving_vertices = self.snapshot_vertices[0].copy()
        self.evolution_index = 1

        self.sub_g = self.graph.subgraph(self.evolving_vertices)
        #self.sub_g.copy_from_parent()
        self._generate_mappings()

    def _generate_mappings(self):
        self.subgraph_to_original_map = utils.from_nn_lib_to_numpy(self.sub_g.ndata[dgl.NID])
        row = numpy.zeros((len(self.subgraph_to_original_map),), dtype=int)
        col = self.subgraph_to_original_map
        data = numpy.arange(len(self.subgraph_to_original_map))
        self.original_to_subgraph_map = utils.sparse1d(scipy.sparse.csc_matrix((data, (row, col))))

    def _generate_snapshot_random(self):
        raise NotImplementedError

    def get_added_vertices(self, delta=None):
        '''
        :param delta: how many past snapshot take into consideration (None=return new vertices added in the last snapshot)
        :return: newly added vertices (ids), labelled or not (boolean)
        '''

        if delta is None:
            delta = 1

        v_set = set()
        for i in range(delta):
            v_set = v_set.union(set(self.snapshot_vertices[self.evolution_index - i - 1]))

        vertices = list(v_set)
        labelled = []

        for v in vertices:
            labelled += [v in self.labelled_vertices]

        return vertices, labelled

    def get_graph(self):
        '''
        :return: current graph
        '''
        return self.sub_g

    def __len__(self):
        '''
        :return: Number of snapshots
        '''
        return len(self.snapshot_vertices)#TODO fix

    def evolve(self):
        '''
        Apply the next snapshot (adding new vertices to the graph)
        :return:
        '''
        self.evolving_vertices += self.snapshot_vertices[self.evolution_index]
        self.evolution_index += 1

        self.sub_g = self.graph.subgraph(self.evolving_vertices)
        self._generate_mappings()

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

        return set(self.snapshot_vertices[self.evolution_index-1]), self.search_depth #search 2 hop from new vertex



