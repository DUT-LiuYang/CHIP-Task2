from keras.layers import *
from Models.BaseModel import BaseModel


class Toymodel(BaseModel):
    def build_model(self):

        word_input_1 = self.Q1 # (?, len)
        word_input_2 = self.Q2  # (?, len)
        sentence_embedding_layer = Embedding(self.embedding_matrix.shape[0],
                                             self.embedding_matrix.shape[1],
                                             weights=[self.embedding_matrix],
                                             input_length=self.word_max_len,
                                             trainable=False,
                                             name='sentence_embedding_layer')
        BGRU1 = Bidirectional(GRU(300, return_sequences=True, dropout=0.2, implementation=2))
        BGRU2 = Bidirectional(GRU(300, return_sequences=False, dropout=0.2, implementation=2))
        sentence_embedding_1 = sentence_embedding_layer(word_input_1)  # (?, len, 300)
        sentence_embedding_2 = sentence_embedding_layer(word_input_2)  # (?, len, 300)

        encoding1 = BGRU1(sentence_embedding_1)
        encoding2 = BGRU1(sentence_embedding_2)

        result1 = BGRU2(encoding1)
        result2 = BGRU2(encoding2)

        mysub = subtract([result1, result2])
        mymut = multiply([result1, result2])
        result = concatenate([result1, result2, mymut, mysub])
        result = Dropout(0.2)(result)
        result = Dense(512, activation='tanh')(result)
        result = Dense(256, activation='tanh')(result)
        predictions = Dense(1, activation='sigmoid')(result)
        # model = Model(inputs=[word_input_1, word_input_2], outputs=predictions)
        # model.summary()
        # model.compile(loss=['binary_crossentropy'],optimizer='Adam', metrics=['accuracy'])

        return predictions

#从basemodel里添加了字符级进行测试
if __name__ == '__main__':
    testmodel=Toymodel()


