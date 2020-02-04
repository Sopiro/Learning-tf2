import tensorflow as tf
import matplotlib.pyplot as plt

(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data()

# Reshape dataset (60000, 28, 28) to (60000, 784)
train_X = train_X.reshape((train_X.shape[0], -1))
test_X = test_X.reshape((test_X.shape[0], -1))

# Normalize pixel data 0~255 to 0.0 to 1.0
train_X = train_X / 255.0
test_X = test_X / 255.0

train_Y = tf.keras.utils.to_categorical(train_Y)
test_Y = tf.keras.utils.to_categorical(test_Y)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=128, kernel_initializer='he_normal', input_shape=(784,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=64, kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=32, kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=10, activation='softmax', kernel_initializer='he_normal'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_X, train_Y, batch_size=32, epochs=10, verbose=1, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_accuracy')])

plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'm-', label='val_accuracy')
plt.plot(history.history['loss'], 'k--', label='loss')
plt.legend()
plt.ylim(0, 1)
plt.show()

print(model.evaluate(test_X, test_Y))
