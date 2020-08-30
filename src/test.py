import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import os

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

# list = ['a', 'b', 'c', 'd']
#
# res = ' '.join(list)
#
# print(res)

# a = [tf.ones((3, 4))]
#
# b = [tf.ones((5, 6))]
#
# c = a + b
#
# l2 = tf.reduce_sum([tf.nn.l2_loss(v) for v in c])
#
# print(l2.numpy())

# a = tf.convert_to_tensor([], dtype=tf.float32)
#
# b = tf.concat([a, tf.convert_to_tensor([2], dtype=tf.float32)], -1)
#
# print(b)
#
# print(b.numpy())

# npa = np.array([10.0])
#
# if os.path.exists('./test.npy'):
#     npa = np.load('./test.npy')
#
# npa = np.append(npa, 1)
#
# print(npa)
#
# npa = np.save('./test.npy', npa)

a = np.arange(3)

a = np.r_[100, a]


print(a)