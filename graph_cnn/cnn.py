from keras.models import Sequential
from keras.layers import Merge, Dense, Convolution2D, MaxPooling2D, Convolution1D, Dropout, Flatten
from keras.utils import np_utils
from graph_cnn.data_util import *


class GraphCnn:
    def __init__(self, w, k, category_num, data, nb_filters=32):
        self.w = w
        self.k = k
        self.category_num = category_num
        self.nb_filters = nb_filters
        self.data = data
        self.model = self.build_model()

    def build_model(self):
        node_conv = Sequential()
        node_conv.add(Convolution2D(self.nb_filters, 1, self.k, border_mode='valid',
                                    input_shape=(1, self.w, self.k), activation='relu'))
        node_conv.add(Convolution2D(self.nb_filters, 1, self.k, border_mode='valid',
                                    input_shape=(1, self.w, self.k), activation='relu'))
        node_conv.add(MaxPooling2D())
        node_conv.add(Dropout(0.25))
        node_conv.add(Flatten())
        node_conv.add(Dense(32, activation='relu'))

        edge_conv = Sequential()
        edge_conv.add(Convolution2D(self.nb_filters, 1, self.k * self.k, border_mode='valid',
                                    input_shape=(1, self.w, self.k * self.k), activation='relu'))
        edge_conv.add(Convolution2D(self.nb_filters, 1, self.k * self.k, border_mode='valid',
                                    input_shape=(1, self.w, self.k * self.k), activation='relu'))
        edge_conv.add(MaxPooling2D())
        edge_conv.add(Dropout(0.25))
        edge_conv.add(Flatten())
        edge_conv.add(Dense(128, activation='relu'))

        merged_layer = Merge(node_conv, edge_conv)

        ret = Sequential(merged_layer)
        ret.add(Dense(64, activation='relu'))
        ret.add(Dropout(0.5))
        ret.add(Dense(self.category_num, activation='softmax'))
        return ret

    def run(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adadelta',
                           metrics=['accuracy'])
        node_x, edge_x, y = self.data.get_xy()
        y = np_utils.to_categorical(y)
        self.model.fit([node_x, edge_x], y, validation_split=0.33, nb_epoch=150, batch_size=10)


def main():
    all_data = load_data()
    dataset = Dataset(all_data['NCI1'], 20, 10)
    cnn = GraphCnn(20, 10, 2, dataset)
    cnn.run()

if __name__ == "__main__":
    main()
