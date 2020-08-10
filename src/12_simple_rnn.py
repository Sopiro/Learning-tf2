import tensorflow as tf
import numpy as np

X = []
Y = []

time_steps = 5

for i in range(6):
    lst = list(range(i, i + time_steps))

    X.append(list(map(lambda c: [c / 10], lst)))
    Y.append((i + 4) / 10)

X = np.array(X)
Y = np.array(Y)

print(X)

# input_shape=[timesteps, input_dimension]
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units=11, activation='tanh', return_sequences=False, input_shape=[time_steps, 1]),
    tf.keras.layers.Dense(units=1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X, Y, epochs=2000, verbose=0)

print(model.predict(X))
