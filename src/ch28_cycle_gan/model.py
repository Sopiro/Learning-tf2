from ch28_cycle_gan.layer import *


# ResNet Generator
class ResNetGenerator(tf.keras.Model):
    def __init__(self, out_channels=3, nker=64, norm='inorm'):
        super(ResNetGenerator, self).__init__()

        self.enc1 = CNR2d(1 * nker, kernel_size=7, stride=1, norm=norm, relu=0.0, padding=3)
        self.enc2 = CNR2d(2 * nker, kernel_size=3, stride=2, norm=norm, relu=0.0)
        self.enc3 = CNR2d(4 * nker, kernel_size=3, stride=2, norm=norm, relu=0.0)

        self.res1 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)
        self.res2 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)
        self.res3 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)
        self.res4 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)
        self.res5 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)
        self.res6 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)
        self.res7 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)
        self.res8 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)
        self.res9 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)

        self.dec1 = DECNR2d(2 * nker, kernel_size=3, stride=2, norm=norm, relu=0.0)
        self.dec2 = DECNR2d(1 * nker, kernel_size=3, stride=2, norm=norm, relu=0.0)
        self.dec3 = CNR2d(out_channels, kernel_size=7, stride=1, norm=None, relu=None, bias=False, padding=3)

    def call(self, x, training=False):
        x = self.enc1(x, training=training)
        x = self.enc2(x, training=training)
        x = self.enc3(x, training=training)

        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        x = self.res3(x, training=training)
        x = self.res4(x, training=training)
        x = self.res5(x, training=training)
        x = self.res6(x, training=training)
        x = self.res7(x, training=training)
        x = self.res8(x, training=training)
        x = self.res9(x, training=training)

        x = self.dec1(x, training=training)
        x = self.dec2(x, training=training)
        x = self.dec3(x, training=training)

        return tf.nn.tanh(x)


# Patch GAN
class Discriminator(tf.keras.Model):
    def __init__(self, out_channels=1, nker=64, norm='inorm', lsgan=True):
        super(Discriminator, self).__init__()

        self.lsgan = lsgan

        self.dsc1 = CNR2d(1 * nker, kernel_size=4, stride=2, norm=None, relu=0.2)
        self.dsc2 = CNR2d(2 * nker, kernel_size=4, stride=2, norm=norm, relu=0.2)
        self.dsc3 = CNR2d(4 * nker, kernel_size=4, stride=2, norm=norm, relu=0.2)
        self.dsc4 = CNR2d(8 * nker, kernel_size=4, stride=1, norm=norm, relu=0.2, padding=1, padding_type='ZERO')
        self.dsc5 = CNR2d(out_channels, kernel_size=4, stride=1, norm=None, relu=None, bias=False, padding=1, padding_type='ZERO')

    def call(self, x, training=False):
        x = self.dsc1(x)
        x = self.dsc2(x)
        x = self.dsc3(x)
        x = self.dsc4(x)
        x = self.dsc5(x)
        if not self.lsgan:
            x = tf.nn.sigmoid(x)

        return x


class UnetGenerator(tf.keras.Model):
    def __init__(self, out_channels=3, nker=64, norm='inorm'):
        super(UnetGenerator, self).__init__()

        self.enc1 = CNR2d(1 * nker, kernel_size=4, stride=2, norm=norm, relu=0.2)
        self.enc2 = CNR2d(2 * nker, kernel_size=4, stride=2, norm=norm, relu=0.2)
        self.enc3 = CNR2d(4 * nker, kernel_size=4, stride=2, norm=norm, relu=0.2)
        self.enc4 = CNR2d(8 * nker, kernel_size=4, stride=2, norm=norm, relu=0.2)
        self.enc5 = CNR2d(8 * nker, kernel_size=4, stride=2, norm=norm, relu=0.2)
        self.enc6 = CNR2d(8 * nker, kernel_size=4, stride=2, norm=norm, relu=0.2)
        self.enc7 = CNR2d(8 * nker, kernel_size=4, stride=2, norm=norm, relu=0.2)
        self.enc8 = CNR2d(8 * nker, kernel_size=4, stride=2, norm=norm, relu=0.0)

        self.dec8 = DECNR2d(8 * nker, kernel_size=4, stride=2, norm=norm, relu=0.0, drop=0.5)
        self.dec7 = DECNR2d(8 * nker, kernel_size=4, stride=2, norm=norm, relu=0.0, drop=0.5)
        self.dec6 = DECNR2d(8 * nker, kernel_size=4, stride=2, norm=norm, relu=0.0, drop=0.5)
        self.dec5 = DECNR2d(8 * nker, kernel_size=4, stride=2, norm=norm, relu=0.0)
        self.dec4 = DECNR2d(4 * nker, kernel_size=4, stride=2, norm=norm, relu=0.0)
        self.dec3 = DECNR2d(2 * nker, kernel_size=4, stride=2, norm=norm, relu=0.0)
        self.dec2 = DECNR2d(1 * nker, kernel_size=4, stride=2, norm=norm, relu=0.0)
        self.dec1 = DECNR2d(1 * out_channels, kernel_size=4, stride=2, norm=None, relu=None, bias=False)

    def call(self, x, training=False):
        enc1 = self.enc1(x, training=training)
        enc2 = self.enc2(enc1, training=training)  # -------------------------------------┐ (bs, 128, 128, 64)
        enc3 = self.enc3(enc2, training=training)  # ------------------------------------┐| (bs, 64, 64, 128)
        enc4 = self.enc4(enc3, training=training)  # -----------------------------------┐|| (bs, 32, 32, 256)
        enc5 = self.enc5(enc4, training=training)  # ----------------------------------┐||| (bs, 16, 16, 512)
        enc6 = self.enc6(enc5, training=training)  # ---------------------------------┐|||| (bs, 8, 8, 512)
        enc7 = self.enc7(enc6, training=training)  # --------------------------------┐||||| (bs, 4, 4, 512)
        enc8 = self.enc8(enc7, training=training)  # -------------------------------┐|||||| (bs, 2, 2, 512)
        dec8 = self.dec8(enc8, training=training)  # (bs, 1, 1, 512)----------------|||||||
        dec7 = self.dec7(tf.concat([enc7, dec8], axis=-1), training=training)  # ---┤|||||| (bs, 2, 2, 1024)
        dec6 = self.dec6(tf.concat([enc6, dec7], axis=-1), training=training)  # ---┼┘||||| (bs, 4, 4, 1024)
        dec5 = self.dec5(tf.concat([enc5, dec6], axis=-1), training=training)  # ---┼-┘|||| (bs, 8, 8, 1024)
        dec4 = self.dec4(tf.concat([enc4, dec5], axis=-1), training=training)  # ---┼--┘||| (bs, 16, 16, 1024)
        dec3 = self.dec3(tf.concat([enc3, dec4], axis=-1), training=training)  # ---┼---┘|| (bs, 32, 32, 512)
        dec2 = self.dec2(tf.concat([enc2, dec3], axis=-1), training=training)  # ---┼----┘| (bs, 64, 64, 256)
        dec1 = self.dec1(tf.concat([enc1, dec2], axis=-1), training=training)  # ---┼-----┘ (bs, 128, 128, 128)
        return tf.nn.tanh(dec1)  # <------------------------------------------------┘(bs, 256, 256, out_channels=3)
