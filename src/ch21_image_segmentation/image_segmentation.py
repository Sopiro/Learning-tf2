import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

dataset, info = tfds.load('oxford_iiit_pet:3.1.0', with_info=True)

# print(info)

train_data_len = info.splits['train'].num_examples
test_data_len = info.splits['test'].num_examples


def load_image(datapoint):
    img = tf.image.resize(datapoint['image'], (128, 128))
    mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    img = tf.cast(img, tf.float32)
    img = img / 255.0
    mask -= 1

    return img, mask


train_dataset = dataset['train'].map(load_image)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(16)

test_dataset = dataset['test'].map(load_image)
test_dataset = test_dataset.repeat()
test_dataset = test_dataset.batch(1)


# plt.figure(figsize=(8, 12))
# for img, mask in train_dataset.take(1):
#     for i in range(3):
#         plt.subplot(3, 2, 2 * i + 1)
#         plt.imshow(img[i])
#
#         plt.subplot(3, 2, 2 * i + 2)
#         plt.imshow(np.squeeze(mask[i], axis=2))
#         plt.colorbar()
#
# plt.show()


def REDNet_seg(num_layers):
    conv_layers = []
    deconv_layers = []
    residual_layers = []

    # Initial settings
    input_layer = tf.keras.layers.Input(shape=(None, None, 3))
    conv_layers.append(tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='same', activation='relu'))

    for i in range(num_layers - 1):
        conv_layers.append(
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, padding='same', activation='relu'))
        deconv_layers.append(
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, padding='same', activation='relu'))

    deconv_layers.append(
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, padding='same', activation='softmax'))

    # Encoder
    x = conv_layers[0](input_layer)

    # Chaining input to output with variable x.
    for i in range(num_layers - 1):
        x = conv_layers[i + 1](x)
        if i % 2 == 0:
            residual_layers.append(x)

    # Decoder
    for i in range(num_layers - 1):
        if i % 2 == 1:
            x = tf.keras.layers.Add()([x, residual_layers.pop()])
            x = tf.keras.layers.Activation('relu')(x)
        x = deconv_layers[i](x)

    x = deconv_layers[-1](x)

    return tf.keras.Model(inputs=input_layer, outputs=x)


model = REDNet_seg(15)
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.summary()

print(model.predict(test_dataset.take(1)).shape)
print(tf.argmax(model.predict(test_dataset.take(1)), -1).shape)

# plt.imshow(np.squeeze(tf.argmax(model.predict(test_dataset.take(1)), axis=-1), 0))
# plt.colorbar()
# plt.show()

# assert False
#
# history = model.fit_generator(train_dataset, epochs=20, steps_per_epoch=train_data_len // 16,
#                               validation_data=test_dataset, validation_steps=test_data_len)

plt.figure(figsize=(12, 12))
for idx, (img, mask) in enumerate(test_dataset.take(3)):
    plt.subplot(3, 3, idx * 3 + 1)
    plt.imshow(img[0])

    plt.subplot(3, 3, idx * 3 + 2)
    plt.imshow(np.squeeze(mask[0], axis=2))

    predict = tf.argmax(model.predict(img), axis=-1)
    plt.subplot(3, 3, idx * 3 + 3)
    plt.imshow(predict[0])
plt.show()

# Test on non resized image
plt.figure(figsize=(12, 12))
for idx, datapoint in enumerate(dataset['test'].shuffle(1).take(3)):
    img = datapoint['image']
    mask = datapoint['segmentation_mask']
    img = tf.cast(img, tf.float32)
    img = img / 255.0
    mask -= 1

    plt.subplot(3, 3, idx * 3 + 1)
    plt.imshow(img)

    plt.subplot(3, 3, idx * 3 + 2)
    plt.imshow(np.squeeze(mask, axis=2))

    predict = tf.argmax(model.predict(tf.expand_dims(img, 0)), -1)
    plt.subplot(3, 3, idx * 3 + 3)
    plt.imshow(predict[0])
plt.colorbar()
plt.show()