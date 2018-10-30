from keras.layers import *
from keras import Model
from keras.layers import Dense, Bidirectional, GRU, Dropout, Lambda, Permute, Dot, Concatenate, Multiply, Add, \
    GlobalAvgPool1D, GlobalMaxPool1D, Embedding
from keras.activations import softmax
import tensorflow as tf
import keras.backend as K
from Models.BaseModel import BaseModel
from All_model.mygru import MYGRU
from run import *






def unchanged_shape(input_shape):
    """Function for Lambda layer"""
    return input_shape

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

def cosine(input_tensor):
     input1=input_tensor[0]
     input2=input_tensor[1]
     print("start")
     print(input1)
     print(input2)

     vec3 = tf.reduce_sum(tf.multiply(input1, input2), axis=2)

     print(vec3)

     vec1 = tf.sqrt(tf.reduce_sum(tf.square(input1), axis=2))
     print(vec1)

     vec2 = tf.sqrt(tf.reduce_sum(tf.square(input2), axis=2))
     print(vec2)


     cosin = vec3 / (vec1 * vec2)
     return cosin

def cos_outshpae(input_shape):
    shape1=list (input_shape[0])
    shape1.pop()
    return tuple (shape1)

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




    def build_model(self):
        BGRU1 = Bidirectional(GRU(300,
                                activation='relu',
                                recurrent_dropout=0.2,
                                return_sequences=True,
                                implementation=2,
                                dropout=0.2))
        BGRU2 =Bidirectional(GRU(300,
                                activation='relu',
                                recurrent_dropout=0.2,
                                return_sequences=False,
                                implementation=2,
                                dropout=0.2))
        encoding1 = BGRU1(self.Q1_emb)#(B,43,512)
        encoding2 = BGRU1(self.Q2_emb)#(B,43,512)
        # result1 = BGRU2(encoding1) #(B,512)
        # result2 = BGRU2(encoding2) #(B,512)
        #
        # mysub = subtract([result1, result2])
        # mymut = multiply([result1, result2])



        att_vec1,att_vec2=soft_attention_alignment(encoding1,encoding2)
        cosval=Lambda(cosine,output_shape=cos_outshpae)([att_vec1,att_vec2])
        res1=BGRU2(att_vec1) #(B,512)
        res2=BGRU2(att_vec2)
        mysub = subtract([res1, res2])
        mymut = multiply([res1, res2])

        result = concatenate([mysub,cosval,mymut])
        result = Dropout(0.5)(result)
        result = Dense(128, activation='tanh')(result)
        result=Dense(32, activation='tanh')(result)


        predictions = Dense(1, activation='sigmoid')(result)



        return predictions

if __name__ == '__main__':
    mymodel=Guo_att_model()
    mymodel.train_model(batch_size=512,epochs=100,kfold_num=5)
