from keras import activations, initializers, regularizers, constraints
from keras.engine import Layer
from keras import backend as K


class CoAttentionLayer(Layer):
    def __init__(self, units,
                 use_bias=True,
                 initializer='glorot_uniform',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(CoAttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.initializer = initializers.get(initializer)
        self.supports_masking = True

    def build(self, input_shape):
        dim = input_shape[0][-1]
        self.kernel = self.add_weight('kernel', [2, dim, self.units],
                                      initializer=self.initializer)
        if self.use_bias:
            self.bias = self.add_weight('bias', [2, self.units],
                                        initializer=initializers.get('zeros'))
        else:
            self.bias = None
        self.built = True

    def projection(self, sentences, i):
        sentences = K.dot(sentences, self.kernel[i])
        if self.use_bias:
            sentences = K.bias_add(sentences, self.bias[i])
        sentences = K.relu(sentences)
        return sentences

    def co_attention(self, Q, E, mask, axis=1):
        mask = K.expand_dims(mask, axis=axis)
        E_beta = E - (1 - mask) * 1e30
        beta_weights = K.softmax(E_beta, axis=3 - axis)
        beta = K.batch_dot(beta_weights, Q, axes=[3 - axis, 1])
        return beta

    def call(self, inputs, mask=None):
        Q1, Q2 = [self.projection(a, i=0) for a in inputs]
        mask1, mask2 = [K.cast(m, dtype=K.dtype(Q1)) for m in mask]

        E = K.batch_dot(Q1, Q2, axes=[2, 2])   # (B, L1, L2)

        # beta
        beta = self.co_attention(inputs[1], E, mask2, axis=1)  # (B, L1, dim)

        # alpha
        alpha = self.co_attention(inputs[0], E, mask1, axis=2)  # (B, L2, dim)

        # intra-attention
        Q1, Q2 = [self.projection(a, i=1) for a in inputs]
        F1 = K.batch_dot(Q1, Q1, axes=[2, 2])  # (B, L1, L1)
        F2 = K.batch_dot(Q2, Q2, axes=[2, 2])  # (B, L2, L2)

        Q1_new = self.co_attention(inputs[0], F1, mask1, axis=1)
        Q2_new = self.co_attention(inputs[1], F2, mask2, axis=1)

        return [beta, alpha, Q1_new, Q2_new]

    def compute_output_shape(self, input_shape):
        return input_shape*2

    def compute_mask(self, inputs, mask=None):
        return mask*2

