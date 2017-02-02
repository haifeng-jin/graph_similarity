import scipy.io
import numpy as np
import pickle


def make_flat(a):
    temp = []
    for elem in a:
        if len(elem):
            temp.append(list(elem[0].flatten()))
        else:
            temp.append([])
    return np.asarray(temp)


def construct_graph(graph, label, filename):
    node_num = len(graph[0][0][0][0])
    if filename in ['DD', 'ENZYMES']:
        edge_labels = []
        adj_list = make_flat(graph[2].flatten())
    elif filename in ['MUTAG']:
        adj_list = make_flat(graph[3])
        temp = graph[2][0][0][0]
        edge_dict = {}
        for line in temp:
            edge_dict[(line[0], line[1])] = line[2]
            edge_dict[(line[1], line[0])] = line[2]
        edge_labels = [[] for i in range(node_num)]
        for u in range(node_num):
            for v in range(node_num):
                if (u + 1, v + 1) in edge_dict.keys():
                    edge_labels[u].append(edge_dict[(u + 1, v + 1)])
    else:
        edge_labels = make_flat(graph[3][0][0][0])
        adj_list = make_flat(graph[2].flatten())
    adj_list = [list(map(lambda y: y - 1, x)) for x in adj_list]
    ret = {'node_labels': graph[0][0][0][0].reshape(node_num),
           'edge_labels': edge_labels,
           'adjacency_list': adj_list,
           # 'adjacency_matrix': graph[1],
           'node_num': node_num,
           'label': label}
    return ret


def get_labels(raw_labels):
    temp = {}
    count = 0
    for label in raw_labels:
        if label not in temp.keys():
            temp[label] = count
            count += 1
    return list(map(lambda x: temp[x], raw_labels))


def load_data(filename):
    mat = scipy.io.loadmat('../' + filename + '.mat')
    ret = []
    labels = get_labels(mat['l' + filename.lower()].flatten())
    for i, graph in enumerate(mat[filename][0]):
        if name in ['DD', 'MUTAG', 'ENZYMES']:
            graph[0], graph[1] = graph[1], graph[0]
        ret.append(construct_graph(graph, labels[i], filename))
    return ret


data_names = ['NCI1', 'NCI109', 'DD', 'MUTAG', 'ENZYMES']
new_file = {}
for index, name in enumerate(data_names):
    pickle.dump(load_data(name), open('../data/' + name + '.graph', 'wb'))

test_file = pickle.load(open('../data/NCI1.graph', 'rb'))
print(len(test_file))
