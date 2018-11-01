from keras import activations, initializers, regularizers, constraints
from keras.engine import Layer
from keras import backend as K


class HermitianDot(Layer):
    def __init__(self,
                 activation=None,
                 use_bias=True,
                 use_bilinear=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(HermitianDot, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.use_bilinear = use_bilinear
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[0][-1]

        self.kernel = self.add_weight(shape=(input_dim, input_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bilinear is False and self.use_bias:
            self.bias = self.add_weight(shape=(input_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, mask=None):
        # Q1: (B, L1, dim), Q2: (B, L2, dim), mask1: (B, L1), mask2: (B, L2)
        Q1, Q2 = inputs
        mask1, mask2 = mask
        if self.use_bilinear:  # bilinear operation
            res = K.dot(Q1, self.kernel)  # (B, L1, dim)
            if self.activation is not None:
                res = self.activation(res)
            output = K.batch_dot(res, Q2, axes=[2, 2])
        else:
            # 求虚部
            res1 = K.dot(Q1, self.kernel)  # (B, L1, dim)
            res2 = K.dot(Q2, self.kernel)  # (B, L2, dim)
            if self.use_bias:
                res1 = K.bias_add(res1, self.bias)
                res2 = K.bias_add(res2, self.bias)
            res1 = K.relu(res1)
            res2 = K.relu(res2)

            # hermitian dot
            out1 = K.batch_dot(Q1, Q2, axes=[2, 2])  # (B, L1, L2)
            out2 = K.batch_dot(res1, res2, axes=[2, 2])

            output = out1 - out2

        score = output

        mask1 = K.expand_dims(K.cast(mask1, K.dtype(Q1)), axis=2)
        mask2 = K.expand_dims(K.cast(mask2, K.dtype(Q2)), axis=1)

        score1 = score - (1 - mask1) * 1e30
        score2 = score - (1 - mask2) * 1e30

        alpha1 = K.softmax(score1, axis=1)
        alpha2 = K.softmax(score2, axis=2)

        Q1_ = K.batch_dot(alpha2, Q2, axes=[2, 1])
        Q2_ = K.batch_dot(alpha1, Q1, axes=[1, 1])

        Q1 = K.concatenate([Q1, Q1_], axis=2)
        Q2 = K.concatenate([Q2, Q2_], axis=2)

        return [Q1, Q2]

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[0])
        output_shape[-1] *= 2
        return [tuple(output_shape)]*2

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(HermitianDot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))