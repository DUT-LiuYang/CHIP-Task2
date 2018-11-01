from keras.constraints import unit_norm

from Models.BaseModel import BaseModel
from keras.layers import *
from keras import backend as K

from layers.HermitianLayer import HermitianDot
from layers.selfattention import SelfAttention


class HermitianCA(BaseModel):
    def build_model(self):
        Q1, Q2 = self.Q1_emb, self.Q2_emb

        bigru = Bidirectional(GRU(256, return_sequences=True, dropout=0.2))
        Q1 = bigru(Q1)
        Q2 = bigru(Q2)

        Q1, Q2 = HermitianDot()([Q1, Q2])

        sa = SelfAttention(1)
        Q1_vector = sa(Q1)
        Q2_vector = sa(Q2)

        vec = [Q1_vector, Q2_vector,
               subtract([Q1_vector, Q2_vector]),
               multiply([Q1_vector, Q2_vector])]

        info = Dense(256, activation='tanh')(concatenate(vec))
        res = Dense(1, activation='hard_sigmoid')(info)

        return res
