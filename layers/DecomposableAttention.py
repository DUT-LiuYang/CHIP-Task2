from keras import backend as K
from keras.engine import Layer


class DecomposableAttention(Layer):
    def __init__(self, use_bias=False, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.supports_masking = False
        self.use_bias = use_bias

    def build(self, input_shape):
        input_shape = input_shape[0]
        dim = input_shape[-1]
        self.kernel1 = self.add_weight(name='kernel1',
                                       shape=[dim, dim],
                                       initializer='glorot_uniform')
        self.kernel2 = self.add_weight(name='kernel2',
                                       shape=[dim, 1],
                                       initializer='glorot_uniform')
        self.built = True

    def call(self, inputs, mask=None):
        Q1, Q2 = inputs

        Q1_ = K.expand_dims(Q1, axis=2)  # (B, L1, 1, dim)
        Q2_ = K.expand_dims(Q2, axis=1)  # (B, 1, L2, dim)

        Q12s = Q1_ - Q2_

        matrix = K.dot(K.abs(Q12s), self.kernel1)
        matrix = K.tanh(matrix)
        matrix = K.dot(matrix, self.kernel2)
        similarity_matrix = K.squeeze(matrix, axis=-1)

        return similarity_matrix

    def compute_output_shape(self, input_shape):
        shape1, shape2 = input_shape

        B, L1, dim1 = shape1
        B, L2, dim2 = shape2
        return tuple((B, L1, L2))
