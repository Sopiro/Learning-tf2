import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np

(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data()

train_X = train_X / 255.0
test_X = test_X / 255.0

train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=(2, 2), activation='elu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=(2, 2), activation='elu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='elu'),  # Latent vector
    tf.keras.layers.Dense(7 * 7 * 64, activation='elu'),
    tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
    tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=(2, 2), padding='same', activation='elu'),
    tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=2, strides=(2, 2), padding='same', activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(train_X, train_X, batch_size=256, epochs=20)

plt.figure(figsize=(4, 8))
for c in range(4):
    plt.subplot(4, 2, c * 2 + 1)
    index = random.randint(0, test_X.shape[0])
    plt.imshow(test_X[index].reshape(28, 28), cmap='gray')
    plt.axis('off')

    plt.subplot(4, 2, c * 2 + 2)
    img = model.predict(np.expand_dims(test_X[index], axis=0))
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()

model.evaluate(test_X, test_X)
