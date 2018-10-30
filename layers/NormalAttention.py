from keras.engine import Layer, InputSpec
from keras import backend as K, activations, initializers, regularizers, constraints
from keras.layers import Dense


class NormalAttention(Dense):
    def __init__(self, **kwargs):
        super(NormalAttention, self).__init__(units=1, **kwargs)
        self.supports_masking = True
        self.input_spec = [self.input_spec, InputSpec(min_ndim=2)]

    def build(self, input_shape):
        input_shape = input_shape[1]
        B, L, dim = input_shape
        super(NormalAttention, self).build((B, L, 3 * dim))
        self.input_spec = [InputSpec(min_ndim=2, axes={-1: dim})] * 2

    def call(self, inputs, mask=None):
        Q1, Q2S = inputs
        _, L, dim = K.int_shape(Q2S)
        Q1_mask, Q2S_mask = mask
        assert Q1_mask is None and Q2S_mask is not None
        Q1_expand = K.repeat(Q1, L)  # (B, L, dim)
        Q2S_mask = K.cast(Q2S_mask, K.dtype(Q2S))

        # vector = K.concatenate([Q1_expand, Q2S, Q1_expand * Q2S, Q1_expand - Q2S], axis=-1)  # (B, L, 2dim)
        vector = K.concatenate([Q1_expand, Q2S, Q1_expand * Q2S], axis=-1)  # (B, L, 2dim)
        score = super(NormalAttention, self).call(vector)  # (B, L, 1)
        score = K.squeeze(score, axis=-1)  # (B, L)
        score -= (1 - Q2S_mask) * 1e30
        alpha = K.softmax(score, axis=-1)
        Q2 = K.batch_dot(alpha, Q2S, axes=[1, 1])
        return Q2

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        return None
