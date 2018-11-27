import keras
from keras.layers import Dense, Bidirectional, GRU, Dropout, Lambda, Permute, Dot, Concatenate, Multiply, Add, \
    GlobalAvgPool1D, GlobalMaxPool1D, Embedding

from All_model.Bi_GRU2_based_Model import LiuModel1
from layers.DecomposableAttention import DecomposableAttention
from keras.activations import softmax
import keras.backend as K
import csv
import numpy as np

from preprocess.csv_reader import CsvReader
from preprocess.example_reader import ExampleReader


def sum_unchanged_shape(input_shape):
    """Function for Lambda layer"""
    return input_shape[0]


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
    # out_ = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=sum_unchanged_shape)([input_1, input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    # sub = substract(input_1, input_2)
    # out_= Concatenate()([sub, mult])
    # return out_
    return mult


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
        self.name = "ESIM_dot_c_extra"

    def build_model(self):

        word_encoding_layer1 = Bidirectional(GRU(300,
                                                 return_sequences=True,
                                                 dropout=0.2))

        encoded_sentence_1 = word_encoding_layer1(self.Q1_emb)  # (?, len, 600)
        encoded_sentence_2 = word_encoding_layer1(self.Q2_emb)  # (?, len, 600)

        # q1_aligned, q2_aligned = minus_soft_attention_alignment(encoded_sentence_1, encoded_sentence_2)
        q1_aligned, q2_aligned = soft_attention_alignment(encoded_sentence_1, encoded_sentence_2)

        q1_combined = Concatenate()([encoded_sentence_1, q2_aligned, submult(encoded_sentence_1, q2_aligned)])
        q2_combined = Concatenate()([encoded_sentence_2, q1_aligned, submult(encoded_sentence_2, q1_aligned)])

        word_encoding_layer2 = Bidirectional(GRU(300,
                                                 return_sequences=True,
                                                 dropout=0.2))
        word_q1_compare = word_encoding_layer2(q1_combined)
        word_q2_compare = word_encoding_layer2(q2_combined)

        word_q1_rep = apply_multiple(word_q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        word_q2_rep = apply_multiple(word_q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        char_encoding_layer1 = Bidirectional(GRU(300,
                                                 return_sequences=True,
                                                 dropout=0.2))

        encoded_sentence_1 = char_encoding_layer1(self.Q1_char_emb)  # (?, len, 600)
        encoded_sentence_2 = char_encoding_layer1(self.Q2_char_emb)  # (?, len, 600)

        q1_aligned, q2_aligned = minus_soft_attention_alignment(encoded_sentence_1, encoded_sentence_2)
        # q1_aligned, q2_aligned = soft_attention_alignment(encoded_sentence_1, encoded_sentence_2)

        q1_combined = Concatenate()([encoded_sentence_1, submult(encoded_sentence_1, q2_aligned)])
        q2_combined = Concatenate()([encoded_sentence_2, submult(encoded_sentence_2, q1_aligned)])

        char_encoding_layer2 = Bidirectional(GRU(300,
                                                 return_sequences=True,
                                                 dropout=0.2))
        q1_combined = char_encoding_layer2(q1_combined)  # (?, len, 600)
        q2_combined = char_encoding_layer2(q2_combined)  # (?, len, 600)

        char_q1_rep = apply_multiple(q1_combined, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        char_q2_rep = apply_multiple(q2_combined, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        merged = Concatenate()([word_q1_rep, word_q2_rep, char_q1_rep, char_q2_rep])
        # merged = Concatenate()([char_q1_rep, char_q2_rep])
        # merged = Concatenate()([word_q1_rep, word_q2_rep])

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
                                       weights=[self.char_embedding_matrix],
                                       trainable=self.args.char_trainable)
            Q1_char_emb = char_embedding(self.Q1_char)
            Q2_char_emb = char_embedding(self.Q2_char)
            embedded += [Q1_char_emb, Q2_char_emb]
        else:
            embedded += [None, None]

        return embedded

    def predict(self):
        # results = self.model.predict([self.test_word_inputs1, self.test_word_inputs2, self.test_char_inputs1, self.test_char_inputs2], batch_size=128, verbose=1)
        results = self.model.predict([self.train_word_inputs1, self.train_word_inputs2, self.train_char_inputs1, self.train_char_inputs2], batch_size=128, verbose=1)
        self.write_results2file(results,
                                file="../results/stacking/train/result1_83.68.csv")

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

    def load_data(self):

        r_dir = self.args.r_dir
        csv_reader = CsvReader(r_dir)
        print("read data from train.csv...")
        train_data, self.train_label = csv_reader.read_csv(name="train.csv", train=True)

        extra_train_data, extra_train_label = csv_reader.read_csv(name="ext_train.csv", train=True)

        print("\nread data from test.csv...")
        test_data, _ = csv_reader.read_csv(name="test.csv", train=False)

        print("\nget word ids - index dic...")
        embedding_file = "word_embedding.txt"
        new_embedding_file = self.args.word_emb_dir
        word_id_index, word_unk = csv_reader.get_ids_from_embeddings(embedding_file, new_embedding_file)  # 9647

        print("\nget char ids - index dic...")
        embedding_file = "char_embedding.txt"
        new_embedding_file = self.args.char_emb_dir
        char_id_index, char_unk = csv_reader.get_ids_from_embeddings(embedding_file, new_embedding_file)  # 2307

        print("\nread question and convert the word id and char id to index using word/char ids - index dic...")
        id_question_words, id_question_chars = csv_reader.read_questions(name="question_id.csv",
                                                                         word_id_index=word_id_index,
                                                                         char_id_index=char_id_index,
                                                                         word_unk=word_unk,
                                                                         char_unk=char_unk)

        er = ExampleReader(r_dir)
        self.embedding_matrix = er.get_embedding_matrix(self.word_embedding_dir)
        if self.args.need_word_level:
            self.train_word_inputs1, self.train_word_inputs2 = er.question_pairs2question_inputs(inputs=train_data, id_questions=id_question_words, max_len=self.word_max_len)
            extra_train_word_inputs1, extra_train_word_inputs2 = er.question_pairs2question_inputs(inputs=extra_train_data, id_questions=id_question_words, max_len=self.word_max_len)
            self.train_word_inputs1 = np.concatenate([self.train_word_inputs1, extra_train_word_inputs1])
            self.train_word_inputs2 = np.concatenate([self.train_word_inputs2, extra_train_word_inputs2])
            self.test_word_inputs1, self.test_word_inputs2 = er.question_pairs2question_inputs(inputs=test_data, id_questions=id_question_words, max_len=self.word_max_len)

        if self.args.need_char_level:
            self.char_embedding_matrix = er.get_embedding_matrix(self.char_embedding_dir)
            self.train_char_inputs1, self.train_char_inputs2 = er.question_pairs2question_inputs(inputs=train_data, id_questions=id_question_chars, max_len=self.char_max_len)
            extra_train_char_inputs1, extra_train_char_inputs2 = er.question_pairs2question_inputs(inputs=extra_train_data, id_questions=id_question_chars, max_len=self.char_max_len)
            self.train_char_inputs1 = np.concatenate([self.train_char_inputs1, extra_train_char_inputs1])
            self.train_char_inputs2 = np.concatenate((self.train_char_inputs2, extra_train_char_inputs2))
            self.test_char_inputs1, self.test_char_inputs2 = er.question_pairs2question_inputs(inputs=test_data, id_questions=id_question_chars, max_len=self.char_max_len)

        self.train_label = np.concatenate([self.train_label, extra_train_label])


if __name__ == '__main__':
    lm2 = LiuModel2()
    lm2.train_model(epochs=12, batch_size=64, kfold_num=10)
    # model = "0.8368246714410011_[0.40855384762607405]_7_ESIM_dot_c_extra"
    # lm2.read_model(model)
    # lm2.predict()
