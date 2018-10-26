import keras
from keras.layers import Dense, Bidirectional, GRU, Dropout, Lambda, Permute, Dot, Concatenate, Multiply, Add, \
    GlobalAvgPool1D, GlobalMaxPool1D
from All_model.Bi_GRU2_based_Model import LiuModel1
from keras.activations import softmax


def unchanged_shape(input_shape):
    """Function for Lambda layer"""
    return input_shape


def soft_attention_alignment(input_1, input_2):
    """Align text representation with neural soft attention"""

    attention = Dot(axes=-1)([input_1, input_2])

    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                              output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_= Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


class LiuModel2(LiuModel1):

    def __init__(self):
        super(LiuModel2, self).__init__()

    def build_model(self):
        encoding_layer1 = Bidirectional(GRU(300,
                                            return_sequences=True,
                                            dropout=0.2))
        encoded_sentence_1 = encoding_layer1(self.Q1_emb)  # (?, len, 600)
        encoded_sentence_2 = encoding_layer1(self.Q2_emb)  # (?, len, 600)

        q1_aligned, q2_aligned = soft_attention_alignment(encoded_sentence_1, encoded_sentence_2)

        q1_combined = Concatenate()([encoded_sentence_1, q2_aligned, submult(encoded_sentence_1, q2_aligned)])
        q2_combined = Concatenate()([encoded_sentence_2, q1_aligned, submult(encoded_sentence_2, q1_aligned)])

        encoding_layer2 = Bidirectional(GRU(300,
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
