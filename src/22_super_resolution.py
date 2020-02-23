import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt

# Download BSD-500 dataset
# 200 for train, 100 for validation, 200 for test
tf.keras.utils.get_file('C:/Users/Sopiro/.keras/datasets/bsd/bsd_images.zip', 'http://bit.ly/35pHZlC', extract=True)

image_root = pathlib.Path('C:/Users/Sopiro/.keras/datasets/bsd/images')

all_image_paths = list(image_root.glob('*/*'))
# print(all_image_paths[0])

train_path, valid_path, test_path = [], [], []

for image_path in all_image_paths:
    r = str(image_path)

    if r.split('.')[-1] != 'jpg':
        continue

    if r.split('\\')[-2] == 'train':
        train_path.append(r)
    elif r.split('\\')[-2] == 'val':
        valid_path.append(r)
    else:
        test_path.append(r)


# hr and lr stands for high res. and low res., respectively
def get_hr_and_lr(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    hr = tf.image.random_crop(img, [50, 50, 3])
    lr = tf.image.resize(hr, [25, 25])
    lr = tf.image.resize(lr, [50, 50])
    return lr, hr


batch_size = 16
train_ds = tf.data.Dataset.list_files(train_path)
train_ds = train_ds.map(get_hr_and_lr)
train_ds = train_ds.repeat()
train_ds = train_ds.batch(batch_size)

valid_ds = tf.data.Dataset.list_files(valid_path)
valid_ds = valid_ds.map(get_hr_and_lr)
valid_ds = valid_ds.repeat()
valid_ds = valid_ds.batch(1)


def REDNet(num_layers):
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
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, padding='same', activation='relu'))

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


def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


# Define REDNet-30
model = REDNet(15)
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='mse', metrics=[psnr_metric])

# Visualize the network
# tf.keras.utils.plot_model(model)

history = model.fit_generator(train_ds, epochs=200, steps_per_epoch=len(train_path) // batch_size,
                              validation_data=valid_ds, validation_steps=len(valid_path), verbose=2)

# Test on BSD image
img = tf.io.read_file(test_path[0])
img = tf.image.decode_jpeg(img, channels=3)
hr = tf.image.convert_image_dtype(img, tf.float32)

down_scale = 2

lr = tf.image.resize(hr, [hr.shape[0] // down_scale, hr.shape[1] // down_scale])
lr = tf.image.resize(lr, [hr.shape[0], hr.shape[1]])

predict_hr = model.predict(np.expand_dims(lr, 0))

print(tf.image.psnr(np.squeeze(predict_hr, 0), hr, max_val=1.0))
print(tf.image.psnr(lr, hr, max_val=1.0))

plt.figure(figsize=(16, 4))

plt.subplot(1, 3, 1)
plt.imshow(hr)
plt.title('original - hr')

plt.subplot(1, 3, 2)
plt.imshow(lr)
plt.title('lr')

plt.subplot(1, 3, 3)
plt.imshow(np.squeeze(predict_hr, 0))
plt.title('sr')

plt.show()

# Test on Set-5 butterfly image
image_path = tf.keras.utils.get_file('butterfly.png', 'http://bit.ly/2oAOxgH')
img = tf.io.read_file(image_path)
img = tf.image.decode_jpeg(img, channels=3)
hr = tf.image.convert_image_dtype(img, tf.float32)

lr = tf.image.resize(hr, [hr.shape[0] // down_scale, hr.shape[1] // down_scale])
lr = tf.image.resize(lr, [hr.shape[0], hr.shape[1]])

predict_hr = model.predict(np.expand_dims(lr, 0))
print(tf.image.psnr(np.squeeze(predict_hr, 0), hr, max_val= 1.0))
print(tf.image.psnr(lr, hr, max_val=1.0))

plt.figure(figsize=(16, 4))

plt.subplot(1, 3, 1)
plt.imshow(hr)
plt.title('original - hr')

plt.subplot(1, 3, 2)
plt.imshow(lr)
plt.title('lr')

plt.subplot(1, 3, 3)
plt.imshow(np.squeeze(predict_hr, 0))
plt.title('sr')

plt.show()