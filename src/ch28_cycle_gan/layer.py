import tensorflow as tf
import tensorflow_addons as tfa


class CNR2d(tf.keras.Model):
    def __init__(self, filters, kernel_size, stride, padding=None, norm='inorm', relu=0.0, bias=True, drop=0.0, padding_type='same'):
        super().__init__()

        layers = []
        if padding is not None:
            if padding_type.lower() == 'zero':
                layers += [Padding(padding=padding, padding_type='CONSTANT', constant=0)]
            else:
                layers += [Padding(padding=padding)]

            padding_type = 'valid'

        layers += [tf.keras.layers.Conv2D(filters=filters,
                                          kernel_size=kernel_size,
                                          strides=stride,
                                          padding=padding_type,
                                          use_bias=bias,
                                          kernel_initializer=tf.keras.initializers.random_normal(0.0, 0.02))]

        if norm is not None:
            if norm == 'bnorm':
                layers += [tf.keras.layers.BatchNormalization()]
            elif norm == 'inorm':
                layers += [tfa.layers.InstanceNormalization()]

        if relu is not None:
            if relu > 0.0:
                layers += [tf.keras.layers.LeakyReLU(relu)]
            else:
                layers += [tf.keras.layers.ReLU()]

        if drop is not None and drop > 0.0:
            layers += [tf.keras.layers.Dropout(drop)]

        self.cnr2d = tf.keras.Sequential(layers)

    def call(self, x, training=False):
        return self.cnr2d(x, training=training)


class DECNR2d(tf.keras.Model):
    def __init__(self, filters, kernel_size, stride, norm='inorm', relu=0.0, bias=True, drop=0.0, padding_type='same'):
        super().__init__()

        layers = []
        layers += [tf.keras.layers.Conv2DTranspose(filters=filters,
                                                   padding=padding_type,
                                                   kernel_size=kernel_size,
                                                   strides=stride,
                                                   use_bias=bias,
                                                   kernel_initializer=tf.keras.initializers.random_normal(0.0, 0.02))]

        if norm is not None:
            if norm == 'bnorm':
                layers += [tf.keras.layers.BatchNormalization()]
            elif norm == 'inorm':
                layers += [tfa.layers.InstanceNormalization()]

        if relu is not None:
            if relu > 0.0:
                layers += [tf.keras.layers.LeakyReLU(relu)]
            else:
                layers += [tf.keras.layers.ReLU()]

        if drop is not None and drop > 0.0:
            layers += [tf.keras.layers.Dropout(drop)]

        self.decnr2d = tf.keras.Sequential(layers)

    def call(self, x, training=False):
        return self.decnr2d(x, training=training)


class ResBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, stride, padding, norm='inorm', relu=0.0, drop=0.0, bias=True):
        super().__init__()

        layers = []
        layers += [CNR2d(filters, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, relu=relu, bias=bias)]

        if drop is not None and drop > 0.0:
            layers += [tf.keras.layers.Dropout(drop)]

        layers += [CNR2d(filters, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, relu=None, bias=bias)]

        self.resblk = tf.keras.Sequential(layers)

    def call(self, x, training=False):
        return x + self.resblk(x, training=training)


class Padding(tf.keras.Model):
    def __init__(self, padding, padding_type='REFLECT', constant=0):
        super(Padding, self).__init__()
        self.padding = padding
        self.padding_type = padding_type
        self.constant = constant

    def call(self, x):
        return tf.pad(x, [[0, 0], [self.padding, self.padding], [self.padding, self.padding], [0, 0]], mode=self.padding_type, constant_values=self.constant)
