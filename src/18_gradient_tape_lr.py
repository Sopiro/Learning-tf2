import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

X = [0.3, -0.78, 1.26, 0.03, 1.11, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
Y = [12.27, 14.44, 11.87, 18.75, 17.52, 16.37, 19.788, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

optimizer = tf.keras.optimizers.Adam(learning_rate=0.2)


@tf.function
def train(v):
    with tf.GradientTape() as tape:
        tape.watch(v)
        pred = v[0] * X + v[1]
        loss = tf.reduce_mean((Y - pred) ** 2)

    grad = tape.gradient(loss, v)
    tf.print(grad, output_stream=sys.stdout)
    optimizer.apply_gradients(grads_and_vars=[(grad, v)])


v = tf.Variable(tf.random.uniform((2,)))

for i in range(1000):
    train(v)

line_x = np.arange(min(X), max(X), 0.01)
line_y = v[0] * line_x + v[1]

plt.plot(line_x, line_y, 'r-')
plt.plot(X, Y, 'bo')
plt.xlabel("Population Growth Rate(%)")
plt.ylabel("Elderly Population Rate(%)")
plt.show()
