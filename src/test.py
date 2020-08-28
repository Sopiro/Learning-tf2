import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt

# tf.keras.backend.set_floatx('float64')

# w1 = tf.keras.layers.Dense(512)
# w2 = tf.keras.layers.Dense(512)
#
# a = np.ones([240, 64, 789])
# b = np.ones([240, 1, 512])
#
# aa = w1(a)
# bb = w2(b)
#
# print(aa.shape)
#
# c = aa + bb
#
# print(c.shape)

# d = np.ones([3, 64, 256])
#
# print(d)
#
# e = tf.reduce_sum(d, axis=1)
#
# print(e)

# f = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
# g = np.array([[[7, 8]], [[9, 10]], [[11, 12]]])
#
# h = [f, g]
#
# print(h)
#
# i = tf.concat(h, axis=-1)
#
# print(i)

# a = tf.constant([[1], [0], [0]], tf.float32)
#
# print(a)
#
# b = tf.nn.softmax(a, axis=0)
#
# print(b)

# for i in range(10):
#     p = tf.random.categorical(([[1., 2., 1., 2., 4., 5., 6., 36., 53., 643., 5., 3., 6., 567., 58., 56., 5., 64.]]), 1)
#     print(p)
#     print(tf.argmax([1., 2., 1., 2., 4., 5., 6., 36., 53., 643., 5., 3., 6., 567., 58., 56., 5., 64.]))

list = ['a', 'b', 'c', 'd']

res = ' '.join(list)

print(res)