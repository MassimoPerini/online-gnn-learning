import numpy as np

'''
Represents a dynamic graph. Supports edge or vertex timestamps
'''

class DynamicGraph:
    '''
    Dynamic Graph interface
    '''

    def __init__(self, graph, snapshots, labelled_vertices, search_depth):
        '''
        Assume label == -1 -> no label
        :param graph: DGL graph
        :param snapshots: n. snapshots (>=1)
        :param labels: vertex labels (numpy column vector)
        :param search_depth: how many <search_depth>-hop neighbours are affected by an update
        '''
        self.graph = graph
        self.snapshots = snapshots
        self.search_depth = search_depth
        self.evolution_index = 0


        assert(snapshots > 0)
        self.labelled_vertices = labelled_vertices #set(np.argwhere(labels != -1)[:,0])


    def get_labelled_vertices(self):
        return self.labelled_vertices

    def get_added_vertices(self):
        raise NotImplementedError

    def get_graph(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def evolve(self):
        raise NotImplementedError
