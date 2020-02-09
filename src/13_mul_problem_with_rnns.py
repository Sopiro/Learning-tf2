import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

for i in range(3000):
    lst = np.random.rand(100)
    idx = np.random.choice(100, 2, replace=False)

    zeros = np.zeros(100)
    zeros[idx] = 1
    X.append(np.array(list(zip(zeros, lst))))
    Y.append(np.prod(lst[idx]))

X = np.array(X)
Y = np.array(Y)

# print(X[0], Y[0])
# assert False

# you can also test on LSTM, SimpleRNN
# by changing   (GRU)↓  to what you want
model = tf.keras.Sequential([
    tf.keras.layers.GRU(units=30, return_sequences=True, input_shape=[100, 2]),
    tf.keras.layers.GRU(units=30),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(X[:2560], Y[:2560], batch_size=32, epochs=100, validation_split=0.2)

plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

model.evaluate(X[:2560], Y[:2560])
prediction = model.predict(X[2560:2565])

for i in range(5):
    print(Y[2560 + i], '\t', prediction[i][0], '\tdiff:', abs(prediction[i][0] - Y[2560 + i]))

prediction = model.predict(X[2560:])

fail = 0

for i in range(len(prediction)):
    if abs(prediction[i][0] - Y[2560+i]) > 0.04:
        fail += 1

print('corectness:', (440 - fail) / 440 * 100, '%')
