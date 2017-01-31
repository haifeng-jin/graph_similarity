import scipy.io


def print_average(filename):
    mat = scipy.io.loadmat(filename + '.mat')
    sum = 0
    edge_sum = 0
    for graph in mat[filename][0]:
        sum += len(graph[0][0][0][0])
    print('The average number of nodes in', filename, 'is', round(sum / len(mat[filename][0])), '.')


def print_average2(filename):
    mat = scipy.io.loadmat(filename + '.mat')
    sum = 0
    edge_sum = 0
    for graph in mat[filename][0]:
        sum += len(graph[1][0][0][0])
    print('The average number of nodes in', filename, 'is', round(sum / len(mat[filename][0])), '.')

print_average('NCI1')
print_average('NCI109')
print_average2('DD')
print_average2('MUTAG')
print_average2('ENZYMES')
