import keras
from keras.layers import Dense, Bidirectional, GRU, Dropout, Lambda, Permute, Dot, Concatenate, Multiply, Subtract, Add, \
    GlobalAvgPool1D, GlobalMaxPool1D
from keras.activations import softmax, tanh
import keras.backend as K
import numpy as np
from layers.Repeat import RepeatVector
from All_model.ESIM import LiuModel2, apply_multiple, unchanged_shape


def weighted(x):
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(x[0])  # (?, 43, 43)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                              output_shape=unchanged_shape)(x[0]))  # (?, 43, 43)
    in1_aligned = Dot(axes=1)([w_att_1, x[1]])  # (?, 43, 600)
    in2_aligned = Dot(axes=1)([w_att_2, x[2]])  # (?, 43, 600)

    return in1_aligned, in2_aligned


def squeeze_output_shape(input_shape):
    return [input_shape[0], input_shape[1], input_shape[2]]


def multiway_soft_attention_alignment(input_1, input_2, max_len, dim):
    """Align text representation with neural soft attention"""

    # ----- Bilinear attention ----- #
    attention = Dot(axes=-1)([input_1,
                              Dense(dim)(input_2)])
    bilinear_in1_aligned, bilinear_in2_aligned = weighted([attention, input_1, input_2])
    # ----- Bilinear attention ----- #

    x1 = RepeatVector(n=max_len, axis=2, shape=[-1, max_len, dim])(input_1)
    x2 = RepeatVector(n=max_len, axis=1, shape=[-1, max_len, dim])(input_2)

    # ----- Minus attention ----- #
    attention = Subtract()([x1, x2])
    attention = Dense(int(dim / 2), activation='tanh')(attention)
    attention = Dense(1)(attention)
    print(np.shape(attention))
    attention = Lambda(lambda x: K.squeeze(x, axis=-1), output_shape=squeeze_output_shape)(attention)
    print(np.shape(attention))
    minus_in1_aligned, minus_in2_aligned = weighted([attention, input_1, input_2])
    # ----- Minus attention ----- #

    # ----- Dot attention ----- #
    attention = Multiply()([x1, x2])
    attention = Dense(int(dim / 2), activation='tanh')(attention)
    attention = Dense(1)(attention)
    attention = Lambda(lambda x: K.squeeze(x, axis=-1), output_shape=squeeze_output_shape)(attention)
    dot_in1_aligned, dot_in2_aligned = weighted([attention, input_1, input_2])
    # ----- Dot attention ----- #

    # ----- Concat attention ----- #
    v1 = Dense(int(dim / 2))(x1)   # (?, 43, 43, dim / 2)
    v2 = Dense(int(dim / 2))(x2)   # (?, 43, 43, dim / 2)
    attention = Lambda(lambda x: tanh(x), output_shape=unchanged_shape)(Add()([v1, v2]))   # (?, 43, 43, dim / 2)
    attention = Dense(1)(attention)   # (?, 43, 43, 1)
    print(np.shape(attention))
    attention = Lambda(lambda x: K.squeeze(x, axis=-1), output_shape=squeeze_output_shape)(attention)
    print(np.shape(attention))
    concat_in1_aligned, concat_in2_aligned = weighted([attention, input_1, input_2])
    # ----- Concat attention ----- #

    in1_aligned = Concatenate()([dot_in1_aligned, bilinear_in1_aligned, minus_in1_aligned, concat_in1_aligned])
    in2_aligned = Concatenate()([dot_in2_aligned, bilinear_in2_aligned, minus_in2_aligned, concat_in2_aligned])

    return in1_aligned, in2_aligned


class LiuModel4(LiuModel2):

    def __init__(self):
        super(LiuModel2, self).__init__()

    def build_model(self):
        encoding_layer1 = Bidirectional(GRU(256,
                                            return_sequences=True,
                                            dropout=0.2))
        encoded_sentence_1 = encoding_layer1(self.Q1_emb)  # (?, len, 600)
        encoded_sentence_2 = encoding_layer1(self.Q2_emb)  # (?, len, 600)

        q1_aligned, q2_aligned = multiway_soft_attention_alignment(encoded_sentence_1, encoded_sentence_2,
                                                                   max_len=self.word_max_len, dim=512)

        q1_combined = Concatenate()([encoded_sentence_1, q2_aligned])
        q2_combined = Concatenate()([encoded_sentence_2, q1_aligned])

        encoding_layer2 = Bidirectional(GRU(256,
                                            return_sequences=True,
                                            dropout=0.2))
        q1_compare = encoding_layer2(q1_combined)
        q2_compare = encoding_layer2(q2_combined)

        q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        merged = Concatenate()([q1_rep, q2_rep])

        dense = Dense(600, activation='elu')(merged)
        dense = Dropout(rate=0.5)(dense)
        predictions = Dense(1, activation='sigmoid')(dense)

        return predictions


if __name__ == '__main__':
    lm4 = LiuModel4()
    lm4.train_model(epochs=50, batch_size=64, kfold_num=5)
