from Models.BaseModel import BaseModel
from keras.layers import *


class IM(BaseModel):
    def build_model(self):
        BGRU1 = Bidirectional(GRU(300, return_sequences=True,
                                  dropout=0.2, implementation=2))
        BGRU2 = Bidirectional(GRU(300, return_sequences=False,
                                  dropout=0.2, implementation=2))

        encoding1 = BGRU1(self.Q1_emb)
        encoding2 = BGRU1(self.Q2_emb)

        result1 = BGRU2(encoding1)
        result2 = BGRU2(encoding2)

        mysub = subtract([result1, result2])
        mymut = multiply([result1, result2])
        result = concatenate([result1, result2, mymut, mysub])
        result = Dropout(0.2)(result)
        result = Dense(512, activation='tanh')(result)
        result = Dense(256, activation='tanh')(result)
        predictions = Dense(1, activation='sigmoid')(result)

        return predictions
