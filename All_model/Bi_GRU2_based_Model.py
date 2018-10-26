import keras
from keras.layers import Dense, Bidirectional, GRU, Dropout, Lambda, BatchNormalization
from run import parse_args
import argparse
from utils import PRF, print_metrics
from Models.BaseModel import BaseModel
from keras.optimizers import Adam


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
        args.need_word_level = True
        args.word_trainable = False
        args.lr = 0.001

        args.save_dir = "../saved_models/"
        args.word_emb_dir = "../instances/word_embed.txt"
        args.char_emb_dir = "../instances/char_embed.txt"
        args.r_dir = "../resource/"

        super(LiuModel1, self).__init__(args)

    def build_model(self):
        encoding_layer1 = Bidirectional(GRU(300,
                                            return_sequences=True,
                                            dropout=0.2))
        encoded_sentence_1 = encoding_layer1(self.Q1_emb)  # (?, len, 600)
        encoded_sentence_2 = encoding_layer1(self.Q2_emb)  # (?, len, 600)

        encoding_layer2 = Bidirectional(GRU(300,
                                            return_sequences=False,
                                            dropout=0.2))

        encoded_sentence_1 = encoding_layer2(encoded_sentence_1)
        encoded_sentence_2 = encoding_layer2(encoded_sentence_2)

        x1 = keras.layers.multiply([encoded_sentence_1, encoded_sentence_2])
        x2 = Lambda(difference,
                    output_shape=no_change)([encoded_sentence_1, encoded_sentence_2])

        x = keras.layers.concatenate([encoded_sentence_1, encoded_sentence_2, x1, x2])
        x = Dense(600, activation='elu')(x)
        x = Dropout(rate=0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)

        return predictions

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
            print_metrics(metrics,
                          metrics_type=self.model.__class__.__name__ + self.args.selfname,
                          save_dir='../logs')


if __name__ == '__main__':
    lm1 = LiuModel1()
    lm1.train_model(epochs=50, batch_size=64, kfold_num=5)
