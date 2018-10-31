from keras.layers import *
from keras import Model
from keras.layers import Dense, Bidirectional, GRU, Dropout, Lambda, Permute, Dot, Concatenate, Multiply, Add, \
    GlobalAvgPool1D, GlobalMaxPool1D, Embedding
from keras.activations import softmax
import tensorflow as tf
import keras.backend as K
from Models.BaseModel import BaseModel
from layers.DRCN_LSTM import DRCN_GRU
from run import *
lstm_unit=512


#并不好用的DCRN





#
def unchanged_shape(input_shape):
    """Function for Lambda layer"""
    return input_shape

def repeat_shape(input_shape):
    myshape=list(input_shape)
    myshape.append(lstm_unit)
    print(myshape)
    return tuple(myshape)



def soft_attention_alignment(input_1, input_2):
    """Align text representation with neural soft attention"""

    attention = Dot(axes=-1)([input_1, input_2])

    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                              output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned

def cosine(input_dense):
     input1=input_dense[0]
     input2=input_dense[1]

     vec3 = tf.reduce_sum(tf.multiply(input1, input2), axis=2)

     print(vec3.shape)

     vec1 = tf.sqrt(tf.reduce_sum(tf.square(input1), axis=2))
     print(vec1)

     vec2 = tf.sqrt(tf.reduce_sum(tf.square(input2), axis=2))
     print(vec2)
     cosin = vec3/(vec1*vec2)
     return  cosin

def cos_outshpae(input_shape):
    shape1=list (input_shape[0])
    shape1.pop()
    return tuple (shape1)


def cos_att(input1,input2,cosinval):

    att_weigth=Lambda(lambda x:softmax(x,axis=1),output_shape=unchanged_shape)(cosinval)
    att_weigth=Lambda(lambda x:K.repeat(x,n=lstm_unit))(att_weigth)
    att_weigth=Permute([2,1])(att_weigth)
    vec1=Multiply()([input1,att_weigth])
    vec2=Multiply()([input2,att_weigth])
    vec1=Lambda(lambda x:K.cumsum(x,axis=1),output_shape=unchanged_shape)(vec1)
    vec2=Lambda(lambda x:K.cumsum(x,axis=1),output_shape=unchanged_shape)(vec2)
    return vec1,vec2


def auto_encoder(input1,input2,dimension):
    encoder1=Dense(256,activation='tanh')(input1)
    encoder1=Dense(128,activation='tanh')(encoder1)
    encoder1=Dense(256,activation='tanh')(encoder1)
    encoder1=Dense(dimension,activation='tanh')(encoder1)

    encoder2=Dense(256,activation='tanh')(input2)
    encoder2=Dense(128,activation='tanh')(encoder2)
    encoder2=Dense(256,activation='tanh')(encoder2)
    encoder2=Dense(dimension,activation='tanh')(encoder2)

    return encoder1,encoder2

class Guo_att_model(BaseModel):

    def __init__(self):
            args = parse_args()
            args.optimizer = 'adam'
            args.loss = 'binary_crossentropy'
            args.need_char_level = False
            args.lr = 0.001

            args.save_dir = "../saved_models/"
            args.word_emb_dir = "../instances/word_embed.txt"
            args.char_emb_dir = "../instances/char_embed.txt"
            args.r_dir = "../resource/"
            super(Guo_att_model, self).__init__(args)


    def embedded(self):
        if self.args.need_word_level:
            shape = self.embedding_matrix.shape
            word_embedding = Embedding(shape[0], shape[1],
                                       mask_zero=False,
                                       weights=[self.embedding_matrix],
                                       trainable=self.args.word_trainable)
            Q1_emb = word_embedding(self.Q1)
            Q2_emb = word_embedding(self.Q2)
            embedded = [Q1_emb, Q2_emb]
        else:
            embedded = [None, None]

        if self.args.need_char_level:
            shape = self.char_embedding_matrix.shape
            char_embedding = Embedding(*shape, mask_zero=True,
                                       weights=[self.char_embedding_matrix], trainable=self.args.char_trainable)
            Q1_char_emb = char_embedding(self.Q1_char)
            Q2_char_emb = char_embedding(self.Q2_char)
            embedded += [Q1_char_emb, Q2_char_emb]
        else:
            embedded += [None, None]

        return embedded

    def build_model(self):
        BGRU1 = Bidirectional(GRU(256,
                                activation='tanh',
                                recurrent_dropout=0.2,
                                return_sequences=True,
                                implementation=2,
                                dropout=0.2))
        BGRU2 =Bidirectional(GRU(256,
                                activation='tanh',
                                recurrent_dropout=0.2,
                                return_sequences=True,
                                implementation=2,
                                dropout=0.2))
        BGRU3 =Bidirectional(GRU(256,
                                activation='tanh',
                                recurrent_dropout=0.2,
                                return_sequences=True,
                                implementation=2,
                                dropout=0.2))



        encoding1 = BGRU1(self.Q1_emb)#(B,43,512)
        encoding2 = BGRU1(self.Q2_emb)#(B,43,512)

        cosine_val=Lambda(cosine,output_shape=cos_outshpae)([encoding1,encoding2])
        cos_v1,cos_v2=cos_att(encoding1,encoding2,cosine_val)

        concat1_v1=Concatenate()([encoding1,cos_v1])
        concat1_v2=Concatenate()([encoding2,cos_v2])

        #autoencoder
        concat1_v1,concat1_v2=auto_encoder(concat1_v1,concat1_v2,dimension=1024)

        step2_encoding1=BGRU2(concat1_v1)
        step2_encoding2=BGRU2(concat1_v2)

        cosine_val_2=Lambda(cosine,output_shape=cos_outshpae)([step2_encoding1,step2_encoding2])
        cos_v1_2,cos_v2_2=cos_att(step2_encoding1,step2_encoding2,cosine_val_2)

        # concat2_v1=Concatenate()([encoding2,cos_v1_2,concat1_v1])
        # concat2_v2=Concatenate()([encoding2,cos_v2_2,concat1_v1])
        #
        # concat2_v1,concat2_v2=auto_encoder(concat2_v1,concat2_v2,dimension=2048)
        #
        # step3_encoding1=BGRU3(concat2_v1)
        # step3_encoding2=BGRU3(concat2_v2)
        #
        # cosine_val_3=Lambda(cosine,output_shape=cos_outshpae)([step3_encoding1,step3_encoding2])
        # cos_v1_3,cos_v2_3=cos_att(step3_encoding1,step3_encoding2,cosine_val_3)


        res1_max=GlobalMaxPool1D()(cos_v1_2)
        res2_max=GlobalMaxPool1D()(cos_v2_2)

        res1_avg=GlobalAveragePooling1D()(cos_v1_2)
        res2_avg=GlobalAveragePooling1D()(cos_v2_2)

        max_sub = subtract([res1_max, res2_max])
        max_mut = multiply([res1_max, res2_max])

        avg_sub = subtract([res1_avg, res2_avg])
        avg_mut = multiply([res1_avg, res2_avg])

        result = Concatenate()([res1_max,res2_max,res1_avg,res2_avg,max_sub,max_mut,avg_sub,avg_mut])
        result = Dropout(0.5)(result)
        result=Dense(1024, activation='tanh')(result)
        result=Dense(256, activation='tanh')(result)
        predictions = Dense(1, activation='sigmoid')(result)



        return predictions

if __name__ == '__main__':
    mymodel=Guo_att_model()
    mymodel.train_model(batch_size=512,epochs=100,kfold_num=5)
