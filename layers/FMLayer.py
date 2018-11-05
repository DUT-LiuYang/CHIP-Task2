from keras import activations, initializers, regularizers, constraints
from keras.engine import Layer
from keras import backend as K


class FMLayer(Layer):
    def __init__(self, k=5,
                 initializer='glorot_uniform',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FMLayer, self).__init__(**kwargs)
        self.k = k
        self.initializer = initializers.get(initializer)
        self.supports_masking = True

    def build(self, input_shape):
        fm_k = self.k
        fm_p = input_shape[-1]

        self.fm = self.add_weight("fm", [1 + fm_p], initializer=self.initializer)
        self.fm_w0 = self.fm[0]
        self.fm_w = self.fm[1:]
        self.fm_V = self.add_weight(shape=(fm_k, fm_p),
                                    initializer=self.initializer,
                                    name='fm_V')

        self.built = True

    def call(self, inputs, mask=None):
        # Q1: (B, L1, dim)
        seq_lens, dims = K.int_shape(inputs)[1:]
        inputs = K.reshape(inputs, [-1, dims])

        fm_linear_terms = self.fm_w0 + K.tf.matmul(inputs, K.expand_dims(self.fm_w, 1))

        fm_interactions_part1 = K.tf.matmul(inputs, K.tf.transpose(self.fm_V))
        fm_interactions_part1 = K.tf.pow(fm_interactions_part1, 2)

        fm_interactions_part2 = K.tf.matmul(K.tf.pow(inputs, 2), K.tf.transpose(K.tf.pow(self.fm_V, 2)))

        fm_interactions = fm_interactions_part1 - fm_interactions_part2

        latent_dim = fm_interactions
        fm_interactions = K.tf.reduce_sum(fm_interactions, 1, keepdims=True)
        fm_interactions = K.tf.multiply(0.5, fm_interactions)
        fm_prediction = K.tf.add(fm_linear_terms, fm_interactions)

        fm_prediction = K.tf.reshape(fm_prediction, [-1, seq_lens, 1])
        latent_dim = K.tf.reshape(latent_dim, [-1, seq_lens, self.k])
        return [fm_prediction, latent_dim]

    def compute_output_shape(self, input_shape):
        B, L, _ = list(input_shape)
        return [tuple((B, L, 1)), tuple((B, L, self.k))]

    def compute_mask(self, inputs, mask=None):
        return mask

