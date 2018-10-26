import keras
from keras.layers import Dense, Bidirectional, GRU, Dropout, Lambda, BatchNormalization
from run import parse_args
import argparse
from Models.BaseModel import BaseModel


def difference(x):
    return x[0] - x[1]


def no_change(input_shape):
    return input_shape[0]


class LiuModel1(BaseModel):

    def __init__(self):

        self.resource_dir = "../resource/"
        self.instance_dir = "../instances/"

        args = parse_args()

        args.optimizer = 'adam'
        args.loss = 'binary_crossentropy'
        args.need_char_level = False
        args.lr = 0.01

        args.save_dir = "../saved_models/"
        args.word_emb_dir = "../instances/word_embed.txt"
        args.char_emb_dir = "../instances/char_embed.txt"
        args.r_dir = "../resource/"

        super(LiuModel1, self).__init__(args)

    def build_model(self):
        encoding_layer1 = Bidirectional(GRU(256,
                                        activation='relu',
                                        recurrent_dropout=0.2,
                                        return_sequences=True,
                                        dropout=0.2))
        encoded_sentence_1 = encoding_layer1(self.Q1_emb)  # (?, len, 512)
        encoded_sentence_2 = encoding_layer1(self.Q2_emb)  # (?, len, 512)

        encoding_layer2 = Bidirectional(GRU(256,
                                            activation='relu',
                                            recurrent_dropout=0.2,
                                            return_sequences=False,
                                            dropout=0.2))

        encoded_sentence_1 = encoding_layer2(encoded_sentence_1)
        encoded_sentence_2 = encoding_layer2(encoded_sentence_2)

        x1 = keras.layers.multiply([encoded_sentence_1, encoded_sentence_2])
        x2 = Lambda(difference,
                    output_shape=no_change)([encoded_sentence_1, encoded_sentence_2])

        x = keras.layers.concatenate([encoded_sentence_1, encoded_sentence_2, x1, x2])
        x = Dense(512, activation='elu')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)

        return predictions


if __name__ == '__main__':
    lm1 = LiuModel1()
    lm1.train_model(epochs=50, batch_size=32, kfold_num=5)
