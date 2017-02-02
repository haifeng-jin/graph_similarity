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


def construct_graph(graph, label):
    node_num = len(graph[0][0][0][0])
    ret = {'node_labels': graph[0][0][0][0].reshape(node_num),
           'edge_labels': make_flat(graph[3][0][0][0]) if len(graph) > 3 else [],
           'adjacency_list': make_flat(graph[2].flatten()),
           # 'adjacency_matrix': graph[1],
           'node_num': node_num,
           'label': label}
    ret['adjacency_list'] = [list(map(lambda y: y - 1, x)) for x in ret['adjacency_list']]
    return ret


def get_labels(raw_labels):
    temp = {}
    count = 0
    for label in raw_labels:
        if label not in temp.keys():
            temp[label] = count
            count += 1
    return list(map(lambda x: temp[x], raw_labels))


def load_data(filename, index_flip):
    mat = scipy.io.loadmat('../' + filename + '.mat')
    ret = []
    labels = get_labels(mat['l' + filename.lower()].flatten())
    for i, graph in enumerate(mat[filename][0]):
        if index_flip:
            graph[0], graph[1] = graph[1], graph[0]
        ret.append(construct_graph(graph, labels[i]))
    print(ret[0]['edge_labels'])
    return ret


data_names = ['NCI1', 'NCI109']
new_file = {}
for index, name in enumerate(data_names):
    flip = False
    if index > 1:
        flip = True
    new_file[name] = load_data(name, flip)

pickle.dump(new_file, open('data', 'wb'))
test_file = pickle.load(open('data', 'rb'))
print(test_file.keys())
