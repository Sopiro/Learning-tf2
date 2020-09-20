import tensorflow as tf
import tensorflow_addons as tfa


class CNR2d(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, stride=1, padding='same', norm='bnorm', relu=0.0, bias=True, drop=0.0):
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

        if relu is not None and relu > 0.0:
            layers += [tf.keras.layers.LeakyReLU(relu)]
        else:
            layers += [tf.keras.layers.ReLU()]

        if drop is not None and drop > 0.0:
            layers += [tf.keras.layers.Dropout(drop)]

        self.cnr2d = tf.keras.Sequential(layers)

    def call(self, x, training=False):
        return self.cnr2d(x, training=training)


class DECNR2d(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, stride=1, padding='same', norm='bnorm', relu=0.0, bias=False, drop=0.0):
        super().__init__()

        layers = []
        layers += [tf.keras.layers.Conv2DTranspose(filters=filters,
                                                   padding=padding,
                                                   kernel_size=kernel_size,
                                                   strides=stride,
                                                   use_bias=bias,
                                                   kernel_initializer='glorot_uniform')]
        if norm is not None:
            if norm == 'bnorm':
                layers += [tf.keras.layers.BatchNormalization()]
            elif norm == 'inorm':
                layers += [tfa.layers.InstanceNormalization()]

        if relu is not None and relu > 0.0:
            layers += [tf.keras.layers.LeakyReLU(relu)]
        else:
            layers += [tf.keras.layers.ReLU()]

        if drop is not None and drop > 0.0:
            layers += [tf.keras.layers.Dropout(drop)]

        self.decnr2d = tf.keras.Sequential(layers)

    def call(self, x, training=False):
        return self.decnr2d(x, training=training)


class ResBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, stride=1, padding='same', norm='inorm', relu=0.0, drop=0.0):
        super().__init__()

        layers = []
        layers += [CNR2d(filters, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, relu=relu)]
        layers += [CNR2d(filters, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, relu=None)]

        if drop is not None and drop > 0.0:
            layers += [tf.keras.layers.Dropout(drop)]

        self.resblk = tf.keras.Sequential(layers)

    def call(self, x, training=False):
        return x + self.resblk(x, training=training)
