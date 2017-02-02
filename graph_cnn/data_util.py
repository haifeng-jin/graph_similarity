import pickle
import math
import numpy


def load_data(filename):
    return pickle.load(open('../data/' + filename + '.graph', 'rb'))


class Graph:
    def __init__(self, graph, w, k):
        self.graph = graph
        self.w = w
        self.k = k
        self.selected = self.select_nodes()
        self.neighbour_matrix = list(map(self.construct_neighbours, self.selected))

    def sort_nodes(self):
        self.graph['degree'] = list(map(lambda elem: len(elem), self.graph['adjacency_list']))
        ret = list(range(self.graph['node_num']))
        ret.sort(key=lambda x: self.graph['degree'][x], reverse=True)
        return ret

    def select_nodes(self):
        stride = int(math.floor((self.graph['node_num'] - 1) / (self.w - 1)))
        sorted_nodes = self.sort_nodes()
        ret = []
        for i in range(self.w):
            ret.append(sorted_nodes[i * stride])
        return ret

    def get_next_layer(self, queue, vis):
        ret = []
        for u in queue:
            for v in self.graph['adjacency_list'][u]:
                if not vis[v]:
                    ret.append(v)
        return ret

    def construct_neighbours(self, a):
        node_num = self.graph['node_num']
        vis = [False] * node_num
        vis[a] = True
        queue = [a]
        ret = []
        while len(ret) < self.k:
            queue.sort(key=lambda x: self.graph['degree'][x], reverse=True)
            ret += queue
            new_queue = self.get_next_layer(queue, vis)
            if not new_queue:
                break
            queue = new_queue
            for i in queue:
                vis[i] = True
        if len(ret) < self.k:
            ret += [-1] * (self.k - len(ret))
        return ret[0:self.k]

    def get_label(self, a):
        if a != -1:
            return self.graph['node_labels'][a]
        return -2

    """
        Return a 3-d tensor (channel, row_num, col_num)
    """
    def node_matrix(self):
        return [[list(map(self.get_label, line)) for line in self.neighbour_matrix]]

    def edge_matrix(self):
        temp = {}
        for i in range(self.graph['node_num']):
            for j in range(self.graph['node_num']):
                temp[(i, j)] = 0
        for u, line in enumerate(self.graph['adjacency_list']):
            for i, v in enumerate(line):
                if self.graph['edge_labels']:
                    label = self.graph['edge_labels'][u][i]
                else:
                    label = 1
                temp[(u, v)] = label
        ret = numpy.ndarray(shape=[self.w, self.k, self.k])
        for i, node in enumerate(self.selected):
            for x, u in enumerate(self.neighbour_matrix[i]):
                if u != -1:
                    for y, v in enumerate(self.neighbour_matrix[i]):
                        if v != -1:
                            ret[i][x][y] = temp[(u, v)]
        return ret.reshape([self.w, self.k * self.k])


class Dataset:
    def __init__(self, data, w, k):
        self.data = data
        self.w = w
        self.k = k
        self.graphs = self.get_graph_list()

    def get_graph_list(self):
        return list(map(lambda x: Graph(x, self.w, self.k), self.data))

    def get_xy(self):
        x_node = list(map(lambda x: x.node_matrix(), self.graphs))
        x_edge = list(map(lambda x: x.edge_matrix(), self.graphs))
        y = list(map(lambda x: x.graph['label'], self.graphs))
        return x_node, x_edge, y
