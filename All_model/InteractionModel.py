from keras.constraints import unit_norm

from Models.BaseModel import BaseModel
from keras.layers import *

from layers.GatedAttention import GatedAttention
from layers.NormalAttention import NormalAttention
from layers.selfattention import SelfAttention
from keras import backend as K, Model
from keras.optimizers import get


def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_true = K.one_hot(K.cast(y_true, 'int32'), 2)
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    y_pred = K.softmax(y_pred, axis=1)
    return K.categorical_crossentropy(y_true, y_pred)


class IM(BaseModel):

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

        self.model = Model(inputs=inputs, outputs=self.output)
        optimizer = get({'class_name': self.args.optimizer, 'config': {'lr': self.args.lr}})
        self.model.compile(optimizer=optimizer, loss=amsoftmax_loss, metrics=['acc'])

    def update_module(self, Q1, Q2):
        bigru = Bidirectional(GRU(256, return_sequences=True,
                              return_state=False,
                              # dropout=self.dropout, recurrent_dropout=self.dropout,
                                          ))
        Q1 = bigru(Q1)
        Q2 = bigru(Q2)
        Q1, Q2 = GatedAttention(units=1)([Q1, Q2])
        return Q1, Q2

    def attention(self, Q1, Q2, implementation=0):
        bigru = Bidirectional(GRU(256, return_sequences=True,
                                  return_state=True,
                                  # dropout=self.dropout, recurrent_dropout=self.dropout,
                                  ))

        Q1, forward_state1, backward_state1 = bigru(Q1)
        Q2, forward_state2, backward_state2 = bigru(Q2)

        last_output1 = merge.concatenate([forward_state1, backward_state1])
        last_output2 = merge.concatenate([forward_state2, backward_state2])
        # print(last_output1, last_output2)
        if implementation == 0:
            Q1 = NormalAttention()([last_output2, Q1])
            Q2 = NormalAttention()([last_output1, Q2])
        elif implementation == 1:
            att = NormalAttention()
            Q1 = att([last_output2, Q1])
            Q2 = att([last_output1, Q2])
        elif implementation == 2:
            Q1 = last_output1
            Q2 = last_output2
        elif implementation == 3:
            return Q1, Q2
        else:
            raise ValueError('implementation value error')

        return Q1, Q2

    def build_model(self):
        update_num = 3

        Q1, Q2 = self.Q1_emb, self.Q2_emb

        for i in range(update_num):
            Q1, Q2 = self.update_module(Q1, Q2)
        Q1, Q2 = self.attention(Q1, Q2, implementation=2)

        # att = SelfAttention(1, activation='tanh')
        # Q1 = att(Q1)
        # Q2 = att(Q2)
        vector = concatenate([  # Q1, Q2,
            merge.multiply([Q1, Q2]),
            # merge.subtract([Q1, Q2], use_abs=True),
            merge.subtract([Q1, Q2]),
            merge.average([Q1, Q2])
        ])
        # vector = merge.subtract([Q1, Q2])
        # vector = merge.add([Q1, Q2])
        # vector = Dropout(self.dropout)(vector)
        # vector = Dense(units=512, activation='tanh')(vector)
        # magic_new = Dense(units=64, activation='tanh')(magic)

        # vector = concatenate([vector, magic_new])
        vector = Dropout(self.args.dropout)(vector)

        vector = Dense(units=256, activation='tanh')(vector)

        vector = Dropout(self.args.dropout)(vector)

        # regression = Dense(units=1, activation='sigmoid')(vector)

        # 这里开始是必写部分，组合上面的compile_model一起写
        x = Lambda(lambda x: K.l2_normalize(x, 1))(vector)
        pred = Dense(2,
                     use_bias=False,
                     kernel_constraint=unit_norm()
                     )(x)
        return pred
