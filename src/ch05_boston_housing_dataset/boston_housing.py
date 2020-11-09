from tensorflow.keras.datasets import boston_housing
import tensorflow as tf
import matplotlib.pyplot as plt

(train_X, train_Y), (test_X, test_Y) = boston_housing.load_data()

# Standardization
x_mean = train_X.mean()
x_std = train_X.std()

train_X -= x_mean
train_X /= x_std
test_X -= x_mean
test_X /= x_std

y_mean = train_Y.mean()
y_std = train_Y.std()

train_Y -= y_mean
train_Y /= y_std
test_Y -= y_mean
test_Y /= y_std

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=52, activation='relu', input_shape=(13,)),
    tf.keras.layers.Dense(units=39, activation='relu'),
    tf.keras.layers.Dense(units=26, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='mse')

model.summary()

'''
validation_split –
Float between 0 and 1. Fraction of the training data to be used as validation data.
The model will set apart this fraction of the training data,
will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
This argument is not supported when `x` is a dataset, generator or `keras.utils.Sequence` instance.

A callback is a set of functions to be applied at given stages of the training procedure.
You can use callbacks to get a view on internal states and statistics of the model during training.
You can pass a list of callbacks (as the keyword argument callbacks)
to the .fit() method of the Sequential or Model classes.
The relevant methods of the callbacks will then be called at each stage of the training.

  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
  # This callback will stop the training when there is no improvement in
  # the validation loss for three consecutive epochs.
  model.fit(data, labels, epochs=100, callbacks=[callback],
      validation_data=(val_data, val_labels))
'''

history = model.fit(train_X, train_Y, batch_size=32, epochs=50, validation_split=0.25, verbose=1
                    , callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')])

plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

print('------------------------------------------------------------------------------------------------------------')
print(model.evaluate(test_X, test_Y))

pred_Y = model.predict(test_X)

plt.figure(figsize=(6, 6))
plt.plot(test_Y, pred_Y, 'b.')
plt.axis([min(test_Y), max(test_Y), min(test_Y), max(test_Y)])

plt.plot([min(test_Y), max(test_Y)], [min(test_Y), max(test_Y)], ls='--', c='.3')
plt.xlabel('test_Y')
plt.ylabel('pred_Y')

plt.show()
