import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd

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

# a = np.arange(3)
#
# a = np.r_[100, a]
#
# print(a)

# a = [tf.ones((100, 1, 300))]
# b = [tf.ones((100, 1, 30))]
#
# c = a + b
#
# print(tf.reduce_sum([tf.size(t) for t in c]).numpy())

# t = tf.zeros((1, 5, 5, 3))
# paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
# # 'constant_values' is 0.
# # rank of 't' is 2.
#
# print(tf.pad(t, paddings, "CONSTANT", constant_values=7))


# a = tf.zeros((3, 5, 5, 3))
# b = tf.ones((3, 5, 5, 3))
#
# print(a.shape)
# print(b.shape)
#
# c = tf.concat([a, b], -1)
#
# print(c.shape)

# folder = 'ch30_image_captioning/flickr30k_images'
#
# data = pd.read_csv(folder + '/results.csv', delimiter='|')
#
# print(data.info())
#
# for a, b, c in data.values:
#     if type(c) != str:
#         print(a, b, c)

# a = np.arange(10)[:, np.newaxis]
# b = np.arange(8)[np.newaxis, :]
#
# c = a * b
# print(c)
#
# c[:, 0::2] = 2
# c[:, 1::2] = 3
#
# print(c)

# a = tf.convert_to_tensor(((1, 2, 3), (4, 5, 6)))
# b = tf.convert_to_tensor(((1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)))
#
# print(tf.matmul(a, b, transpose_b=True))

# a = tf.keras.metrics.Mean()
# b = tf.keras.metrics.Mean()
#
# for i in range(1, 11):
#     a(i)
#     b.update_state(i)
#
# print(a.result())
# print(b.result())

# a = tf.constant([[[2, 20, 30, 3, 6]], [[2, 20, 30, 3, 60]]])
#
# print(a.shape)
#
# b = tf.argmax(a, axis=-1)
#
# print(b)
# print(tf.squeeze(b))

# from ch29_transformer.layers import *
#
# pos_encoding = positional_encoding(100, 100)
# print(pos_encoding.shape)
#
# plt.pcolormesh(pos_encoding[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 100))
# plt.ylim((0, 100))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()

from ch30_transformer_captioning.models import *

learning_rate = CustomSchedule(512)
plt.plot(learning_rate(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()

# loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none', label_smoothing=0.1)
#
#
# def loss_function(real, pred):
#     # real.shape == (batch_size, seq_len)
#     # pred.shape == (batch_size, seq_len, vocab_size)
#     real = tf.one_hot(tf.cast(real, tf.int32), pred.shape[-1])
#
#     loss_ = loss_object(real, pred)
#
#     print(loss_)
#
#     return tf.reduce_sum(loss_)
#
#
# a = tf.convert_to_tensor([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=tf.float32)
# b = tf.convert_to_tensor([[[1, 0], [1, 0], [1, 0]],
#                           [[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]],
#                           [[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]]], dtype=tf.float32)
#
#
# print('result', loss_function(a, b))
