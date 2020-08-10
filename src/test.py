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

print(tf.math.logical_not(tf.math.equal([1, 3, 4, 5, 0, 1, 0, 12, 4, 6], 0)))
