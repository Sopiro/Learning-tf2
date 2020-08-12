import numpy as np
import tensorflow as tf

# a = np.ones([1000, 64, 2048])
#
# b = np.ones([2048, 256])
#
# c = np.dot(a, b)
#
# print(c.shape)
#
# aa = np.random.normal(size=(10,))
#
# print(aa)
# print()
#
# print(aa[-3:])

a = tf.ones(shape=(10,), dtype=tf.float32)
b = tf.random.normal(shape=(10,), dtype=tf.float32)

print(a)
print(b)

a = a + b

print(a)
