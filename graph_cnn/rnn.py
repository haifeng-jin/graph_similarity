from __future__ import print_function

import itertools
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from graph_cnn.data_util import *
from keras.utils import np_utils


# import six.moves.range as range

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


class GraphRnn:
    def __init__(self, data):
        # Parameters for the model and dataset
        TRAINING_SIZE = 50000
        DIGITS = 3
        INVERT = True
        # Try replacing GRU, or SimpleRNN
        RNN = recurrent.LSTM
        HIDDEN_SIZE = 128
        BATCH_SIZE = 128
        LAYERS = 1
        MAXLEN = DIGITS + 1 + DIGITS

        questions = []
        expected = []
        seen = set()

        node_x, edge_x, y = data.get_xy()
        node_x = list(map(lambda x: list(itertools.chain(*x[0])), node_x))
        #print(node_x[0])
        y = np_utils.to_categorical(y)

        model = Sequential()
        # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
        # note: in a situation where your input sequences have a variable length,
        # use input_shape=(None, nb_feature).
        model.add(RNN(HIDDEN_SIZE, input_shape=(len(node_x), len(node_x[0]))))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(node_x, y, validation_split=0.33, nb_epoch=150, batch_size=10)


def main():
    data = load_data('DD')
    dataset = Dataset(data, 20, 10)
    rnn = GraphRnn(dataset)


if __name__ == "__main__":
    main()
