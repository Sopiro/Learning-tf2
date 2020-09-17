import tensorflow as tf
from layer import *


class Generator(tf.keras.Model):

    def __init__(self, out_channels, nker=64, norm='inorm'):
        super(Generator, self).__init__()

        self.enc1 = CNR2d(1 * nker, kernel_size=7, stride=1, padding='same', norm=norm, relu=0.0)
        self.enc2 = CNR2d(2 * nker, kernel_size=3, stride=2, padding='same', norm=norm, relu=0.0)
        self.enc3 = CNR2d(4 * nker, kernel_size=3, stride=2, padding='same', norm=norm, relu=0.0)

        self.res = tf.keras.Sequential()

        for i in range(9):
            self.res.add(ResBlock(4 * nker, kernel_size=3, stride=1, padding='same', norm=norm, relu=0.0))

        self.dec1 = DECNR2d(2 * nker, kernel_size=3, stride=2, padding='same', norm=norm, relu=0.0)
        self.dec2 = DECNR2d(1 * nker, kernel_size=3, stride=2, padding='same', norm=norm, relu=0.0)
        self.dec3 = CNR2d(out_channels, kernel_size=7, stride=1, padding='same', norm=None, relu=None)

    def call(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.res(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)

        return tf.nn.tanh(x)


class Discriminator(tf.keras.Model):
    def __init__(self, out_channels, nker=64, norm='bnorm'):
        super(Discriminator, self).__init__()

        self.enc1 = CNR2d(1 * nker, kernel_size=4, stride=2, padding='same', norm=None, relu=0.2, bias=False)
        self.enc2 = CNR2d(2 * nker, kernel_size=4, stride=2, padding='same', norm=norm, relu=0.2, bias=False)
        self.enc3 = CNR2d(4 * nker, kernel_size=4, stride=2, padding='same', norm=norm, relu=0.2, bias=False)
        self.enc4 = CNR2d(8 * nker, kernel_size=4, stride=2, padding='same', norm=norm, relu=0.2, bias=False)
        self.enc5 = CNR2d(out_channels, kernel_size=4, stride=2, padding='same', norm=None, relu=None, bias=False)

    def call(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        return tf.nn.sigmoid(x)
