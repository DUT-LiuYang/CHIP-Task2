import keras
from keras.layers import Dense, Bidirectional, GRU, Dropout, Lambda, Concatenate, Conv1D
from keras import backend as K
from All_model.ESIM import LiuModel2


class LiuModel3(LiuModel2):

    def __init__(self):
        super(LiuModel2, self).__init__()

    def build_model(self):
        encoding_layer1 = Bidirectional(GRU(300,
                                            return_sequences=True,
                                            dropout=0.2))
        emb1 = encoding_layer1(self.Q1_emb)  # (?, len, 600)
        emb2 = encoding_layer1(self.Q2_emb)  # (?, len, 600)

        conv12 = Conv1D(filters=100, kernel_size=1, padding='same', activation='relu')
        conv22 = Conv1D(filters=100, kernel_size=2, padding='same', activation='relu')
        conv32 = Conv1D(filters=100, kernel_size=3, padding='same', activation='relu')
        conv42 = Conv1D(filters=100, kernel_size=4, padding='same', activation='relu')
        conv52 = Conv1D(filters=50, kernel_size=5, padding='same', activation='relu')
        conv62 = Conv1D(filters=50, kernel_size=6, padding='same', activation='relu')

        conv1a = conv12(emb1)
        conv1b = conv12(emb2)

        conv2a = conv22(emb1)
        conv2b = conv22(emb2)

        conv3a = conv32(emb1)
        conv3b = conv32(emb2)

        conv4a = conv42(emb1)
        conv4b = conv42(emb2)

        conv5a = conv52(emb1)
        conv5b = conv52(emb2)

        conv6a = conv62(emb1)
        conv6b = conv62(emb2)

        mergea = Concatenate()([conv1a, conv2a, conv3a, conv4a, conv5a, conv6a])
        mergeb = Concatenate()([conv1b, conv2b, conv3b, conv4b, conv5b, conv6b])

        encoding_layer2 = Bidirectional(GRU(300,
                                            return_sequences=False,
                                            dropout=0.2))
        q1_compare = encoding_layer2(mergea)
        q2_compare = encoding_layer2(mergeb)

        diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(600,))([q1_compare, q2_compare])
        mul = Lambda(lambda x: x[0] * x[1], output_shape=(600,))([q1_compare, q2_compare])

        merge = Concatenate()([diff, mul, q1_compare, q2_compare])
        x = Dense(600, activation='elu')(merge)
        x = Dropout(rate=0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)

        return predictions


if __name__ == '__main__':
    lm3 = LiuModel3()
    lm3.train_model(30, 64, 5)
