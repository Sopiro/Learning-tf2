import tensorflow as tf
import tensorflow_addons as tfa


class CNR2d(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, stride=1, padding='same', norm='bnorm', relu=0.0, bias=True):
        super().__init__()

        layers = []
        layers += [tf.keras.layers.Conv2D(filters=filters,
                                          kernel_size=kernel_size,
                                          strides=stride,
                                          padding=padding,
                                          use_bias=bias,
                                          kernel_initializer='glorot_uniform')]

        if norm is not None:
            if norm == 'bnorm':
                layers += [tf.keras.layers.BatchNormalization()]
            elif norm == 'inorm':
                layers += [tfa.layers.InstanceNormalization()]

        if relu is not None and relu >= 0.0:
            layers += [tf.keras.layers.ReLU() if relu == 0 else tf.keras.layers.LeakyReLU(relu)]

        self.cnr2d = tf.keras.Sequential(layers)

    def call(self, x):
        return self.cnr2d(x)


class DECNR2d(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, stride=1, padding='same', norm='bnorm', relu=0.0, bias=True):
        super().__init__()

        layers = []
        layers += [tf.keras.layers.Conv2DTranspose(filters=filters,
                                                   padding=padding,
                                                   kernel_size=kernel_size,
                                                   strides=stride,
                                                   use_bias=bias,
                                                   output_padding=1,
                                                   kernel_initializer='glorot_uniform')]
        if norm is not None:
            if norm == 'bnorm':
                layers += [tf.keras.layers.BatchNormalization()]
            elif norm == 'inorm':
                layers += [tfa.layers.InstanceNormalization()]

        if relu is not None and relu >= 0.0:
            layers += [tf.keras.layers.ReLU() if relu == 0 else tf.keras.layers.LeakyReLU(relu)]

        self.decnr2d = tf.keras.Sequential(layers)

    def call(self, x):
        return self.decnr2d(x)


class ResBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, stride=1, padding='same', norm='inorm', relu=0.0):
        super().__init__()

        layers = []
        layers += [CNR2d(filters, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, relu=relu)]
        layers += [CNR2d(filters, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, relu=None)]

        self.resblk = tf.keras.Sequential(layers)

    def call(self, x):
        # return tf.concat(x, -1, self.resblk(x))
        return x + self.resblk(x)
