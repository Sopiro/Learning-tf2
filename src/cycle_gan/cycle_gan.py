import tensorflow as tf
import matplotlib.pyplot as plt
from model import *
import pathlib

# @tf.function
# def train():
# g = Generater(3)
# d = Discriminator(1)
#
# a = tf.ones(shape=(1, 512, 512, 3))
# b = g.call(a)
#
# print(d(b))
#
# b = tf.reshape(b, (512, 512, 3))
#
# # b = tf.multiply(b)
#
# plt.imshow(b)
# plt.show()

path = pathlib.Path('dataset/trainA/')

files = list(path.glob('*.jpg'))


img = tf.io.read_file(str(files[0]))
img = tf.image.decode_jpeg(img, channels=3)

plt.imshow(img)
plt.show()


# nch = 3
#
# g_a2b = Generater(nch, norm='inorm')
# g_b2a = Generater(nch, norm='inorm')
#
# d_a = Discriminator(1, norm='bnorm')
# d_b = Discriminator(1, norm='bnorm')
#
