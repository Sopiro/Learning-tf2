import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

(train_X, _), _ = tf.keras.datasets.mnist.load_data()

# plt.imshow(train_X[0], cmap='gray')
# plt.show()

train_X = train_X[:10000]
train_X = np.reshape(train_X, (-1, 784))
train_X = train_X / 255.0

discriminator = tf.keras.Sequential([
    tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(units=1, activation='linear')
])

generator = tf.keras.Sequential([
    tf.keras.layers.Dense(units=128, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(units=784, activation='tanh')
])  # -1 ~ 1

# discriminator.summary()

optimizer_d = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer_g = tf.keras.optimizers.Adam(learning_rate=0.001)

d_weight = 1e-3
g_weight = 1e1


@tf.function
def train(x, z):
    with tf.GradientTape(persistent=True) as tape:
        loss_d = -(tf.math.log(discriminator(x)) + tf.math.log(1.0 - discriminator(generator(z)))) * d_weight
        loss_g = -tf.math.log(discriminator(generator(z))) * g_weight

    grad_d = tape.gradient(loss_d, discriminator.trainable_variables)
    grad_g = tape.gradient(loss_g, generator.trainable_variables)

    # tf.print(grad_g)

    optimizer_d.apply_gradients(grads_and_vars=zip(grad_d, discriminator.trainable_variables))
    optimizer_g.apply_gradients(grads_and_vars=zip(grad_g, generator.trainable_variables))

    del tape


def get_latent(num=1):
    return tf.random.uniform(shape=(num, 100))
    # return tf.zeros(shape=(num, 100))


def process(img):
    res = tf.reshape(img, (28, 28))
    return res / 2.0 + 0.5


test_img = train_X[0]
test_img = tf.expand_dims(test_img, 0)

latent = get_latent()

plt.figure(figsize=(10, 10))
ran = 1000
wh = ran / 100
for i in range(ran):
    train(test_img, get_latent())

    if i % wh == 0:
        plt.subplot(10, 10, i // wh + 1)
        plt.imshow(process(generator(latent)), cmap='gray')

gen_img = generator(latent)
print('real image:', discriminator(test_img))
print('fake image:', discriminator(gen_img))

plt.show()
