import keras
from keras.engine import Model
from keras.layers import Dense, Bidirectional, GRU, Dropout, Lambda, Permute, Dot, Concatenate, Multiply, Add, \
    GlobalAvgPool1D, GlobalMaxPool1D, Embedding, Subtract, TimeDistributed, Input, BatchNormalization
from sklearn.model_selection import StratifiedKFold, train_test_split

from All_model.Bi_GRU2_based_Model import LiuModel1
from layers.DecomposableAttention import DecomposableAttention
from keras.activations import softmax, tanh
import keras.backend as K
from keras.optimizers import get
import csv
import numpy as np
from preprocess.csv_reader import CsvReader
from preprocess.feature_reader import FeatureReader
from utils import print_metrics, PRF


def unchanged_shape(input_shape):
    """Function for Lambda layer"""
    return input_shape


def weighted(x):
    print(np.shape(x[0]))
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(x[0])  # (?, 43, 43)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                              output_shape=unchanged_shape)(x[0]))  # (?, 43, 43)
    print(np.shape(w_att_1))
    in1_aligned = Dot(axes=1)([w_att_1, x[1]])  # (?, 43, 600)
    in2_aligned = Dot(axes=1)([w_att_2, x[2]])  # (?, 43, 600)

    return in1_aligned, in2_aligned


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


def minus_soft_attention_alignment(input_1, input_2):
    """Align text representation with neural soft attention"""

    attention = DecomposableAttention()([input_1, input_2])

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
        self.name = "ESIM_abs_sub_char_magic"

    def build_model(self):

        # word_encoding_layer1 = Bidirectional(GRU(300,
        #                                          return_sequences=True,
        #                                          dropout=0.2))
        #
        # encoded_sentence_1 = word_encoding_layer1(self.Q1_emb)  # (?, len, 600)
        # encoded_sentence_2 = word_encoding_layer1(self.Q2_emb)  # (?, len, 600)
        #
        # q1_aligned, q2_aligned = minus_soft_attention_alignment(encoded_sentence_1, encoded_sentence_2)
        # # q1_aligned, q2_aligned = soft_attention_alignment(encoded_sentence_1, encoded_sentence_2)
        #
        # q1_combined = Concatenate()([encoded_sentence_1, q2_aligned, submult(encoded_sentence_1, q2_aligned)])
        # q2_combined = Concatenate()([encoded_sentence_2, q1_aligned, submult(encoded_sentence_2, q1_aligned)])
        #
        # word_encoding_layer2 = Bidirectional(GRU(300,
        #                                          return_sequences=True,
        #                                          dropout=0.2))
        # word_q1_compare = word_encoding_layer2(q1_combined)
        # word_q2_compare = word_encoding_layer2(q2_combined)
        #
        # word_q1_rep = apply_multiple(word_q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        # word_q2_rep = apply_multiple(word_q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        char_encoding_layer1 = Bidirectional(GRU(300,
                                                 return_sequences=True,
                                                 dropout=0.2))

        encoded_sentence_1 = char_encoding_layer1(self.Q1_char_emb)  # (?, len, 600)
        encoded_sentence_2 = char_encoding_layer1(self.Q2_char_emb)  # (?, len, 600)

        q1_aligned, q2_aligned = minus_soft_attention_alignment(encoded_sentence_1, encoded_sentence_2)
        # q1_aligned, q2_aligned = soft_attention_alignment(encoded_sentence_1, encoded_sentence_2)

        q1_combined = Concatenate()([encoded_sentence_1, q2_aligned, submult(encoded_sentence_1, q2_aligned)])
        q2_combined = Concatenate()([encoded_sentence_2, q1_aligned, submult(encoded_sentence_2, q1_aligned)])

        char_encoding_layer2 = Bidirectional(GRU(300,
                                                 return_sequences=True,
                                                 dropout=0.2))
        char_q1_compare = char_encoding_layer2(q1_combined)
        char_q2_compare = char_encoding_layer2(q2_combined)

        char_q1_rep = apply_multiple(char_q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        char_q2_rep = apply_multiple(char_q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        # merged = Concatenate()([word_q1_rep, word_q2_rep, char_q1_rep, char_q2_rep])
        self.magic = Input(shape=(4,), dtype='float32', name='magic_input')
        magic_dense = BatchNormalization()(self.magic)
        magic_dense = Dense(64, activation='elu')(magic_dense)
        merged = Concatenate()([char_q1_rep, char_q2_rep, magic_dense])
        # merged = Concatenate()([word_q1_rep, word_q2_rep, magic_dense])

        dense = Dense(600, activation='elu')(merged)
        dense = Dropout(rate=0.5)(dense)
        predictions = Dense(1, activation='sigmoid')(dense)

        return predictions

    def embedded(self):
        if self.args.need_word_level:
            shape = self.embedding_matrix.shape
            word_embedding = Embedding(shape[0], shape[1],
                                       mask_zero=False,
                                       weights=[self.embedding_matrix],
                                       trainable=self.args.word_trainable)
            Q1_emb = word_embedding(self.Q1)
            Q2_emb = word_embedding(self.Q2)
            embedded = [Q1_emb, Q2_emb]
        else:
            embedded = [None, None]

        if self.args.need_char_level:
            shape = self.char_embedding_matrix.shape
            char_embedding = Embedding(*shape, mask_zero=False,
                                       weights=[self.char_embedding_matrix], trainable=self.args.char_trainable)
            Q1_char_emb = char_embedding(self.Q1_char)
            Q2_char_emb = char_embedding(self.Q2_char)
            embedded += [Q1_char_emb, Q2_char_emb]
        else:
            embedded += [None, None]

        return embedded

    def predict(self):
        f = FeatureReader()
        _, magic_b = f.get_magic_feature()
        results = self.model.predict([self.test_word_inputs1, self.test_word_inputs2, self.test_char_inputs1, self.test_char_inputs2, magic_b], batch_size=128, verbose=1)
        self.write_results2file(results,
                                file="../results/result1_85.46_0.358.csv")

    def write_results2file(self, results=[], file=""):

        r_dir = self.args.r_dir
        csv_reader = CsvReader(r_dir)
        test_data, _ = csv_reader.read_csv(name="test.csv", train=False)

        out = open(file, 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        print(str(np.shape(results)))
        csv_write.writerow(['qid1', 'qid2', 'label'])
        for ids, i in zip(test_data, results):
            # if i[0] > 0.5:
            #     csv_write.writerow([ids[0], ids[1], 1])
            # else:
            #     csv_write.writerow([ids[0], ids[1], 0])
            csv_write.writerow([ids[0], ids[1], i[0]])
        out.close()

    def one_train(self, epochs, batch_size,
                  train_data, train_label,
                  dev_data, dev_label):
        self.compile_model()
        for e in range(epochs):
            history = self.model.fit(train_data, train_label, batch_size=batch_size, verbose=2,
                                     validation_data=(dev_data, dev_label))
            dev_out = self.model.predict(dev_data, batch_size=2 * batch_size, verbose=0)
            metrics = PRF(dev_label, (dev_out > 0.5).astype('int32').reshape([-1]))
            metrics['epoch'] = e + 1
            metrics['val_loss'] = history.history['val_loss']
            print_metrics(metrics, metrics_type=self.__class__.__name__ + self.args.selfname,
                          save_dir='../logs')

            file = str(metrics['macro_prf'][2]) + "_" + str(metrics['val_loss']) + "_" + str(e + 1) + "_" + self.name
            self.save_model(file)

    def train_model(self, epochs, batch_size, kfold_num=0):

        f = FeatureReader()
        magic_a, magic_b = f.get_magic_feature()

        inputs = [self.train_word_inputs1, self.train_word_inputs2,
                  self.train_char_inputs1, self.train_char_inputs2, magic_a]
        if kfold_num > 1:
            kfold = StratifiedKFold(n_splits=kfold_num, shuffle=True)
            for train_index, dev_index in kfold.split(self.train_word_inputs1, self.train_label):
                train_data = [data[train_index] for data in inputs if data is not None]
                train_label = self.train_label[train_index]
                dev_data = [data[dev_index] for data in inputs if data is not None]
                dev_label = self.train_label[dev_index]

                self.one_train(epochs, batch_size,
                               train_data, train_label, dev_data, dev_label)

        else:
            inputs = [a for a in inputs if a is not None]
            print([a.shape for a in inputs])
            inputs.append(self.train_label)
            all_data = train_test_split(*inputs, test_size=0.2, random_state=1)
            train_data = [all_data[2*i] for i in range(len(inputs))]
            dev_data = [all_data[2*i + 1] for i in range(len(inputs))]

            self.one_train(epochs, batch_size,
                           train_data[:-1], train_data[-1],
                           dev_data[:-1], dev_data[-1])

    def compile_model(self):

        self.Q1, self.Q2, self.Q1_char, self.Q2_char = self.make_input()
        self.Q1_emb, self.Q2_emb, self.Q1_char_emb, self.Q2_char_emb = self.embedded()
        self.output = self.build_model()

        if self.args.need_word_level:
            inputs = [self.Q1, self.Q2]
        else:
            inputs = []
        if self.args.need_char_level:
            inputs += [self.Q1_char, self.Q2_char]

        inputs += [self.magic]

        self.model = Model(inputs=inputs, outputs=self.output)
        optimizer = get({'class_name': self.args.optimizer, 'config': {'lr': self.args.lr}})
        self.model.compile(optimizer=optimizer, loss=self.args.loss, metrics=['acc'])
        self.model.summary()


if __name__ == '__main__':
    lm2 = LiuModel2()
    # lm2.train_model(epochs=11, batch_size=64, kfold_num=5)
    model = "0.8546531049812561_[0.3584303517341614]_6_ESIM_abs_sub_char_magic"
    lm2.read_model(model)
    lm2.predict()
