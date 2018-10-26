from keras.layers import *
from keras.layers import Dense, Bidirectional, GRU, Dropout, Lambda, BatchNormalization
from Models.BaseModel import BaseModel
from keras import Model
from run import *


class Toymodel(BaseModel):
    def __init__(self):

        args = parse_args()
        args.optimizer = 'adam'
        args.loss = 'binary_crossentropy'
        args.need_char_level = False
        args.lr = 0.01

        args.save_dir = "../saved_models/"
        args.word_emb_dir = "../instances/word_embed.txt"
        args.char_emb_dir = "../instances/char_embed.txt"
        args.r_dir = "../resource/"
        args.need_char_level=True
        super(Toymodel, self).__init__(args)

    def build_model(self):


        BGRU1 = Bidirectional(GRU(256,
                                activation='relu',
                                recurrent_dropout=0.2,
                                return_sequences=True,
                                implementation=2,
                                dropout=0.2))
        BGRU2 =Bidirectional(GRU(256,
                                activation='relu',
                                recurrent_dropout=0.2,
                                return_sequences=False,
                                implementation=2,
                                dropout=0.2))

        Char_BGRU1=Bidirectional(GRU(256,
                                activation='relu',
                                recurrent_dropout=0.2,
                                return_sequences=True,
                                implementation=2,
                                dropout=0.2))
        Char_BGRU2=Bidirectional(GRU(256,
                                activation='relu',
                                recurrent_dropout=0.2,
                                return_sequences=False,
                                implementation=2,
                                dropout=0.2))

        encoding1 = BGRU1(self.Q1_emb)
        encoding2 = BGRU1(self.Q2_emb)

        char_encoding1=Char_BGRU1(self.Q1_char_emb)
        char_encoding2=Char_BGRU1(self.Q2_char_emb)

        result1 = BGRU2(encoding1)
        result2 = BGRU2(encoding2)

        char_res1=Char_BGRU2(char_encoding1)
        char_res2=Char_BGRU2(char_encoding2)


        mysub = subtract([result1, result2])
        mymut = multiply([result1, result2])

        charsub=subtract([char_res1, char_res2])
        charmut=multiply([char_res1, char_res2])



        result = concatenate([result1, result2, mymut, mysub])
        char_res = concatenate([char_res1,char_res2,charsub,charmut])

        result = Dropout(0.2)(result)
        char_res= Dropout(0.2)(char_res)

        char_res=Dense(512, activation='tanh')(char_res)
        char_res=Dense(256, activation='tanh')(char_res)
        result = Dense(512, activation='tanh')(result)
        result = Dense(256, activation='tanh')(result)

        fin_res=concatenate([result,char_res])
        predictions = Dense(1, activation='sigmoid')(fin_res)


        # model = Model(inputs=[self.Q1, self.Q2,self.Q1_char,self.Q2_char], outputs=[predictions])
        # model.summary()
        # model.compile(loss=['binary_crossentropy'],optimizer='Adam', metrics=['accuracy'])

        return predictions

#从basemodel里添加了字符级进行测试
if __name__ == '__main__':
    testmodel=Toymodel()


