import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = [0.3, -0.78, 1.26, 0.03, 1.11, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
Y = [12.27, 14.44, 11.87, 18.75, 17.52, 16.37, 19.788, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=6, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(units=1, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')

# model.summary()

history = model.fit(X, Y, epochs=10)

print(history.history['loss'])

line_x = np.arange(min(X), max(X), 0.01)
line_y = model.predict(line_x)

plt.plot(line_x, line_y, 'r-')
plt.plot(X, Y, 'bo')
# plt.plot(history.history['loss'], 'k--')
plt.xlabel("Population Growth Rate(%)")
plt.ylabel("Elderly Population Rate(%)")
plt.show()
