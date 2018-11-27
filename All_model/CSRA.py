from keras import Model
from keras.backend import epsilon
from keras.backend.tensorflow_backend import _to_tensor

from Models.BaseModel import BaseModel
from keras.layers import *

from layers.CoAttentionLayer import CoAttentionLayer
from layers.FMLayer import FMLayer
from keras.optimizers import get

from layers.MultiHeadAttention import MultiHeadAttention
from layers.selfattention import SelfAttention


class pooling(Layer):
    def __init__(self, **kwargs):
        super(pooling, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        mask = K.expand_dims(K.cast(mask, K.dtype(inputs)))
        inputs -= (1 - mask) * 1e30
        f_m = K.max(inputs, axis=1)

        inputs *= mask
        f_e = K.sum(inputs, axis=1) / K.sum(mask, axis=1)

        return K.concatenate([f_m, f_e], axis=1)

    def compute_output_shape(self, input_shape):
        B, L, dim = list(input_shape)
        return tuple((B, dim * 2))

    def compute_mask(self, inputs, mask=None):
        return None


def new_loss(y_true, y_pred):
    # transform back to logits
    _epsilon = _to_tensor(epsilon(), y_pred.dtype.base_dtype)
    output = K.tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)

    return - y_true * K.pow(1-output, 2) * K.log(output) \
           - (1-y_true) * K.pow(output, 2) * K.log(1-output)


class CSRA(BaseModel):
    # def compile_model(self):
    #
    #     self.Q1, self.Q2, self.Q1_char, self.Q2_char = self.make_input()
    #     self.Q1_emb, self.Q2_emb, self.Q1_char_emb, self.Q2_char_emb = self.embedded()
    #     self.output = self.build_model()
    #
    #     if self.args.need_word_level:
    #         inputs = [self.Q1, self.Q2]
    #     else:
    #         inputs = []
    #     if self.args.need_char_level:
    #         inputs += [self.Q1_char, self.Q2_char]
    #
    #     self.model = Model(inputs=inputs, outputs=self.output)
    #     optimizer = get({'class_name': self.args.optimizer, 'config': {'lr': self.args.lr}})
    #     self.model.compile(optimizer=optimizer, loss=new_loss, metrics=['acc'])

    def build_model(self):
        Q1, Q2 = self.Q1_emb, self.Q2_emb

        bigru = Bidirectional(GRU(256, activation='tanh', return_sequences=True, dropout=0.1))
        Q1 = bigru(Q1)
        Q2 = bigru(Q2)

        mha = MultiHeadAttention(n_head=8, d_model=512, d_k=64, d_v=64, dropout=0.2, mode=0, use_norm=True)
        Q1, _ = mha(Q1, Q1, Q1)
        Q2, _ = mha(Q2, Q2, Q2)
        beta, alpha, Q1_new, Q2_new = CoAttentionLayer(units=256)([Dropout(0.2)(Q1), Dropout(0.2)(Q2)])

        # fm_con = FMLayer(k=5)
        # fm_sub = FMLayer(k=5)
        # fm_mul = FMLayer(k=5)
        pairs = [[beta, Q1], [alpha, Q2], [Q1, Q1_new], [Q2, Q2_new]]
        # concatenate_res = [concatenate(m) for m in pairs]
        subtract_res = [subtract(m) for m in pairs]
        multiply_res = [multiply(m) for m in pairs]
        res = [subtract_res, multiply_res]
        features = [[a for a in m] for m in zip(*res)]

        Q1_E = concatenate([Q1] + features[0] + features[2], axis=-1)
        Q2_E = concatenate([Q2] + features[1] + features[3], axis=-1)

        bigru = Bidirectional(GRU(256, activation='tanh', return_sequences=True, dropout=0.2))
        Q1_vector = bigru(Q1_E)
        Q2_vector = bigru(Q2_E)

        Q1_vector = pooling()(Q1_vector)
        Q2_vector = pooling()(Q2_vector)

        vec = [Q1_vector, Q2_vector,
               subtract([Q1_vector, Q2_vector]),
               multiply([Q1_vector, Q2_vector])]
        vec = concatenate(vec)
        # info = Highway(activation='relu')(vec)
        # info = Highway(activation='relu')(info)
        info = Dense(256, activation='tanh')(Dropout(0.5)(vec))
        res = Dense(1, activation='hard_sigmoid')(info)

        return res
