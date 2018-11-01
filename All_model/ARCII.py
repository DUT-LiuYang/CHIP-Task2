from __future__ import print_function
from __future__ import absolute_import
from keras.layers import *
from All_model.ESIM import LiuModel2

from layers.Match import *
from utils import *


class ARCII(LiuModel2):

    def __init__(self):
        super(ARCII, self).__init__()
        self.kernel_counts_2d = [32, 32]
        self.kernel_sizes_2d = [[3, 3], [3, 3]]
        self.num_conv2d_layers = 2
        self.mpool_sizes_2d = [[3, 3], [3, 3]]

    def build(self):
        encoding_layer1 = Bidirectional(GRU(300,
                                            return_sequences=True,
                                            dropout=0.2))
        encoded_sentence_1 = encoding_layer1(self.Q1_emb)  # (?, len, 600)
        encoded_sentence_2 = encoding_layer1(self.Q2_emb)  # (?, len, 600)

        q_conv1 = Conv1D(32, 3, padding='same')(encoded_sentence_1)
        show_layer_info('Conv1D', q_conv1)
        d_conv1 = Conv1D(32, 3, padding='same')(encoded_sentence_2)
        show_layer_info('Conv1D', d_conv1)

        cross = Match(match_type='plus')([q_conv1, d_conv1])
        show_layer_info('Match-plus', cross)

        z = Reshape((self.word_max_len, self.word_max_len, -1))(cross)
        show_layer_info('Reshape', z)

        for i in range(2):
            z = Conv2D(filters=self.kernel_counts_2d[i], kernel_size=self.kernel_sizes_2d[i], padding='same', activation='relu')(z)
            show_layer_info('Conv2D', z)
            z = MaxPooling2D(pool_size=(self.mpool_sizes_2d[i][0], self.mpool_sizes_2d[i][1]))(z)
            show_layer_info('MaxPooling2D', z)

        pool1_flat = Flatten()(z)
        show_layer_info('Flatten', pool1_flat)
        pool1_flat_drop = Dropout(rate=0.2)(pool1_flat)
        show_layer_info('Dropout', pool1_flat_drop)

        out_ = Dense(1)(pool1_flat_drop)
        show_layer_info('Dense', out_)

        return out_


if __name__ == '__main__':
    ar = ARCII()
    ar.train_model(30, 64, 5)
