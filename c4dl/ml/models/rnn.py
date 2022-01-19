from tensorflow.keras.layers import Layer, Conv2D
import tensorflow as tf
from .blocks import ConvBlock, GRUResBlock
from .layers import Warp

class CustomGateGRU(Layer):
    def __init__(self, 
        update_gate=None, reset_gate=None, output_gate=None,
        return_sequences=False, time_steps=1,
        **kwargs):

        super().__init__(**kwargs)

        self.update_gate = update_gate
        self.reset_gate = reset_gate
        self.output_gate = output_gate
        self.return_sequences = return_sequences
        self.time_steps = time_steps

    def call(self, inputs):
        (xt,h) = inputs

        def step(inputs, states):
            x = inputs
            h = states[0]
            xh = tf.concat((x,h), axis=-1)
            z = self.update_gate(xh)
            r = self.reset_gate(xh)
            o = self.output_gate(tf.concat((x,r*h), axis=-1))
            h = z*h + (1-z)*tf.math.tanh(o)
            return h, [h]

        (last_h, h, _) = tf.keras.backend.rnn(step, xt, [h])

        if self.return_sequences:
            return h
        else:
            return last_h
        

        """
        h_all = []
        for t in range(self.time_steps):
            x = xt[:,t,...]
            xh = tf.concat((x,h), axis=-1)
            z = self.update_gate(xh)
            r = self.reset_gate(xh)
            o = self.output_gate(tf.concat((x,r*h), axis=-1))
            h = z*h + (1-z)*tf.math.tanh(o)
            if self.return_sequences:
                h_all.append(h)

        return tf.stack(h_all,axis=1) if self.return_sequences else h
        """

class ConvGRU(Layer):
    def __init__(self, channels, conv_size=(3,3),
        return_sequences=False, time_steps=1,
        **kwargs):

        super().__init__(**kwargs)

        self.update_gate = Conv2D(channels, conv_size, activation='sigmoid',
            padding='same')
        self.reset_gate = Conv2D(channels, conv_size, activation='sigmoid',
            padding='same')
        self.output_gate = Conv2D(channels, conv_size, padding='same')

        self.return_sequences = return_sequences
        self.time_steps = time_steps

    @tf.function
    def iterate(self, x, h):
        xh = tf.concat((x,h), axis=-1)
        z = self.update_gate(xh)
        r = self.reset_gate(xh)
        o = self.output_gate(tf.concat((x,r*h), axis=-1))
        h = z*h + (1.0-z)*tf.math.tanh(o)
        return h

    def call(self, inputs):
        (xt,h) = inputs

        h_all = []
        for t in range(self.time_steps):
            x = xt[:,t,...]
            h = self.iterate(x,h)
            if self.return_sequences:
                h_all.append(h)

        return tf.stack(h_all,axis=1) if self.return_sequences else h


class ResGRU(ConvGRU):
    def __init__(self, channels, conv_size=(3,3),
        return_sequences=False, time_steps=1,
        **kwargs):

        dropout = kwargs.pop("dropout", 0.0)
        norm = kwargs.pop("norm", None)
        super(ConvGRU, self).__init__(**kwargs)

        self.update_gate = GRUResBlock(channels, conv_size=conv_size,
            final_activation='sigmoid', padding='same', dropout=dropout,
            norm=norm)
        self.reset_gate = GRUResBlock(channels, conv_size=conv_size,
            final_activation='sigmoid', padding='same', dropout=dropout,
            norm=norm)
        self.output_gate = GRUResBlock(channels, conv_size=conv_size,
            final_activation='linear', padding='same', dropout=dropout,
            norm=norm)

        self.return_sequences = return_sequences
        self.time_steps = time_steps


class TrajGRU(ConvGRU):
    def __init__(self, *args, max_warp=8.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.warp = Warp(max_extent=max_warp)

    def call(self, inputs):
        (xt,h) = inputs

        h_all = []
        for t in range(self.time_steps):
            x = xt[:,t,...]
            h = self.warp([h, tf.concat((x,h), axis=-1)])
            xh = tf.concat((x,h), axis=-1)
            z = self.update_gate(xh)
            r = self.reset_gate(xh)
            o = self.output_gate(tf.concat((x,r*h), axis=-1))
            h = z*h + (1-z)*tf.math.tanh(o)
            if self.return_sequences:
                h_all.append(h)

        return tf.stack(h_all,axis=1) if self.return_sequences else h
