import unittest
from graph_cnn.data_util import *


class TestDataUtil(unittest.TestCase):
    graph1 = {'node_num': 3,
              'adjacency_list': [[1], [0, 2], [1]],
              'node_labels': [2, 1, 0],
              'edge_labels': [[1], [1, 1], [1]],
              'label': 1}

    graph2 = {'node_num': 10,
              'adjacency_list': [[1, 2, 3], [0, 4], [0, 5], [0, 6], [1, 7, 8, 9], [2, 7, 8, 9], [3, 7, 8, 9], [4, 5, 6],
                                 [4, 5, 6], [4, 5, 6]],
              'node_labels': [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
              'edge_labels': [[1, 2, 3], [10, 14], [20, 25], [30, 36], [41, 47, 48, 49], [52, 57, 58, 59],
                              [63, 67, 68, 69], [74, 75, 76], [84, 85, 86], [94, 95, 96]],
              'label': 2}

    def test_sort_nodes(self):
        graph = Graph(self.graph1, 2, 4)
        ans = graph.sort_nodes()
        self.assertEqual([1, 0, 2], ans)

    def test_select_nodes(self):
        graph = Graph(self.graph1, 2, 4)
        ans = graph.select_nodes()
        self.assertEqual([1, 2], ans)

    def test_select_more(self):
        graph = Graph(self.graph2, 9, 4)
        self.assertTrue(-1 not in graph.selected)

    def test_select_all(self):
        graph = Graph(self.graph2, 10, 4)
        self.assertTrue(-1 not in graph.selected)

    def test_construct_neighbours(self):
        graph = Graph(self.graph1, 2, 4)
        graph.sort_nodes()
        ans = graph.construct_neighbours(0)
        self.assertEqual([0, 1, 2, -1], ans)

    def test_neighbour_matrix(self):
        graph = Graph(self.graph2, 9, 4)
        self.assertEqual([[4, 7, 8, 9],
                          [5, 7, 8, 9],
                          [6, 7, 8, 9],
                          [0, 1, 2, 3],
                          [7, 4, 5, 6],
                          [8, 4, 5, 6],
                          [9, 4, 5, 6],
                          [1, 4, 0, 7],
                          [2, 5, 0, 7]], graph.neighbour_matrix)

    def test_node_matrix(self):
        graph = Graph(self.graph2, 9, 4)
        self.assertEqual([[5, 2, 1, 0],
                          [4, 2, 1, 0],
                          [3, 2, 1, 0],
                          [9, 8, 7, 6],
                          [2, 5, 4, 3],
                          [1, 5, 4, 3],
                          [0, 5, 4, 3],
                          [8, 5, 9, 2],
                          [7, 4, 9, 2]], graph.node_matrix())

    def test_edge_matrix(self):
        graph = Graph(self.graph2, 9, 4)
        self.assertEqual((9, 16), graph.edge_matrix().shape)

    def test_dataset(self):
        dataset = Dataset([self.graph1, self.graph2], 2, 2)
        self.assertEqual(3, len(dataset.get_xy()))
