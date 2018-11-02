
from Models.BaseModel import BaseModel
from keras.layers import *

from layers.CoAttentionLayer import CoAttentionLayer
from layers.FMLayer import FMLayer
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


class CSRA(BaseModel):
    def build_model(self):
        Q1, Q2 = self.Q1_emb, self.Q2_emb

        bigru = Bidirectional(GRU(256, return_sequences=True, dropout=0.2))
        Q1 = bigru(Q1)
        Q2 = bigru(Q2)

        beta, alpha, Q1_new, Q2_new = CoAttentionLayer(units=256)([Dropout(0.2)(Q1), Dropout(0.2)(Q2)])

        fm_con = FMLayer(k=5)
        fm_sub = FMLayer(k=5)
        fm_mul = FMLayer(k=5)
        pairs = [[beta, Q1], [alpha, Q2], [Q1, Q1_new], [Q2, Q2_new]]
        concatenate_res = [fm_con(concatenate(m)) for m in pairs]
        subtract_res = [fm_sub(subtract(m)) for m in pairs]
        multiply_res = [fm_mul(multiply(m)) for m in pairs]
        res = [concatenate_res, subtract_res, multiply_res]
        features = [[a[0] for a in m] for m in zip(*res)]

        Q1_E = concatenate([Q1] + features[0] + features[2], axis=-1)
        Q2_E = concatenate([Q2] + features[1] + features[3], axis=-1)

        bigru = Bidirectional(GRU(256, return_sequences=True, dropout=0.2))
        Q1 = bigru(Q1_E)
        Q2 = bigru(Q2_E)

        Q1_vector = pooling()(Q1)
        Q2_vector = pooling()(Q2)

        vec = [Q1_vector, Q2_vector,
               subtract([Q1_vector, Q2_vector]),
               multiply([Q1_vector, Q2_vector])]
        vec = concatenate(vec)
        info = Highway(activation='relu')(vec)
        info = Highway(activation='relu')(info)
        res = Dense(1, activation='hard_sigmoid')(info)

        return res
