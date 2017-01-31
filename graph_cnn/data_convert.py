import scipy.io
import numpy as np
import pickle


def make_flat(a):
    temp = []
    for elem in a:
        if elem.size:
            temp.append(list(elem[0].flatten()))
        else:
            temp.append([])
    return np.asarray(temp)


def load_data(filename, flip):
    mat = scipy.io.loadmat('../' + filename + '.mat')
    ret = []
    for graph in mat[filename][0]:
        if flip:
            graph[0], graph[1] = graph[1], graph[0]
        node_num = len(graph[0][0][0][0])
        new_graph = {'node_labels': graph[0][0][0][0].reshape(node_num),
                     'edge_labels': make_flat(graph[3][0][0][0]) if graph.size > 3 else [],
                     'adjacency_list': make_flat(graph[2].flatten()),
                     'adjacency_matrix': graph[1],
                     'node_num': node_num}
        ret.append(new_graph)
    return ret


data_names = ['NCI1', 'NCI109', 'DD', 'MUTAG', 'ENZYMES']
new_file = {}
for index, name in enumerate(data_names):
    flip = False
    if index > 1:
        flip = True
    new_file[name] = load_data(name, flip)

pickle.dump(new_file, open('data', 'wb'))
test_file = pickle.load(open('data', 'rb'))
print(test_file.keys())
