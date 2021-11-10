import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D


class LocalAttention2D(Layer):
    def __init__(self, key_channels, query_channels, value_channels, rad=16):
        self.key_conv = Conv2D(kernel_size=(1,1))
        self.query_conv = Conv2D(kernel_size=(1,1))
        self.value_conv = Conv2D(kernel_size=(1,1))
        self.output_conv = Conv2D(kernel_size=(1,1))
        
        (di,dj) = np.mgrid[-rad:rad+1,-rad:rad+1]
        R_sqr = di**2 + dj**2
        in_range = (R_sqr <= rad**2)
        self.di = di[in_range]
        self.dj = dj[in_range]

    def call(self, x):
        key = self.key_conv(x)
        query = self.query_conv(x)
        value = self.value_conv(x)
        
        terms = []
        for (di,dj) in zip(self.di, self.dj):
            shifted_key = padded_shift(key, di, dj)
            m = tf.tensordot(shifted_key, query, axis=-1)
            shifted_value = padded_shift(value, di, dj)
            terms.append(m * shifted_value)

        o = tf.math.add_n(terms)
        return self.output_conv(o)

    



