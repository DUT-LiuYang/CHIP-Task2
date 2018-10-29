from keras.engine import Layer, InputSpec
import keras.backend as K


class RepeatVector(Layer):

    def __init__(self, n, axis, shape, **kwargs):
        super(RepeatVector, self).__init__(**kwargs)
        self.n = n
        self.axis = axis
        self.shape = shape
        self.input_spec = InputSpec(min_ndim=2)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape.insert(self.axis, self.n)
        return tuple(output_shape)

    def call(self, inputs):
        res = K.repeat_elements(inputs, axis=self.axis, rep=self.n)
        output_shape = self.shape
        output_shape.insert(self.axis, self.n)
        res = K.reshape(res, output_shape)
        return res

    def get_config(self):
        config = {'n': self.n, 'axis': self.axis, 'shape': self.shape}
        base_config = super(RepeatVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
