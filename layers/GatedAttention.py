from keras import backend as K
from keras.engine import Layer


class GatedAttention(Layer):
    def __init__(self, units, use_bias=False, **kwargs):
        super(GatedAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.use_bias = use_bias

    def build(self, input_shape):
        input_shape = input_shape[0]
        dim = input_shape[-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=[3*dim, self.units],
                                      initializer='glorot_uniform')
        # self.W_minus = self.add_weight(name='minus',
        #                                shape=[dim, self.units],
        #                                initializer='glorot_uniform')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer='zero',
                                        name='bias',
                                        regularizer=None,
                                        constraint=None)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        Q1, Q2 = inputs
        # print(Q1, Q2)
        # Q1 = K.reshape(Q1, [-1, 39, K.int_shape(Q1)[-1]])
        # Q2 = K.reshape(Q2, [-1, 39, K.int_shape(Q2)[-1]])
        Q1_mask, Q2_mask = mask  # (B, L1), (B, L2)
        Q1_mask = K.cast(Q1_mask, K.dtype(Q1))
        Q2_mask = K.cast(Q2_mask, K.dtype(Q2))
        Q1_mask = K.expand_dims(Q1_mask)  # (B, L1, 1)
        Q2_mask = K.expand_dims(Q2_mask, axis=1)  # (B, 1, L2)

        Q1_ = K.expand_dims(Q1, axis=2)  # (B, L1, 1, dim)
        Q2_ = K.expand_dims(Q2, axis=1)  # (B, 1, L2, dim)
        Q12m = Q1_ * Q2_  # (B, L1, L2, dim)
        # Q12abs = K.abs(Q1_ - Q2_)
        Q12s = Q1_ - Q2_
        # gate = K.hard_sigmoid(K.dot(Q12S, self.W_minus))
        # Q12s = gate * Q12S + (1 - gate) * Q12abs
        # Q12p = (Q1_ + Q2_) / 2
        one = K.ones_like(Q12m)
        all_for_one = K.concatenate([
                                     Q1_ * one, Q2_ * one,
                                     Q12m,
                                     Q12s,
                                     # Q12p
                                     ])
        all_for_one = K.dot(all_for_one, self.kernel)
        if self.use_bias:
            all_for_one = K.bias_add(all_for_one, self.bias)
        all_for_one = K.tanh(all_for_one)
        print(K.int_shape(all_for_one))
        simularity_matrix = K.squeeze(all_for_one, axis=-1)
        print(K.int_shape(simularity_matrix))
        # simularity_matrix = K.batch_dot(Q1, Q2, axes=[2, 2])  # (B, L1, L2)

        score1 = simularity_matrix - (1 - Q1_mask) * 1e30
        alpha1 = K.softmax(score1, axis=1)
        Q2_ = K.batch_dot(alpha1, Q1, axes=[1, 1])

        score2 = simularity_matrix - (1 - Q2_mask) * 1e30
        alpha2 = K.softmax(score2, axis=2)
        Q1_ = K.batch_dot(alpha2, Q2, axes=[2, 1])
        # print(Q1_, Q2_)
        # Q1 = K.concatenate([Q1, Q1_, Q1 * Q1_])
        # Q2 = K.concatenate([Q2, Q2_, Q2 * Q2_])
        # print(Q1, Q2)
        # print('Q1:', Q1)
        # Q1 = Q1 * Q1_
        # Q2 = Q2 * Q2_
        return [Q1_, Q2_]

    def compute_output_shape(self, input_shape):
        shape1, shape2 = input_shape
        # print(shape1, shape2)
        B, L1, dim1 = shape1
        B, L2, dim2 = shape2
        return [tuple((B, L1, dim1))]*2

    def compute_mask(self, inputs, mask=None):
        return mask

