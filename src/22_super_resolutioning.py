import tensorflow as tf
import pathlib

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


train_ds = tf.data.Dataset.list_files(train_path)
train_ds = train_ds.map(get_hr_and_lr)
train_ds = train_ds.repeat()
train_ds = train_ds.batch(16)

valid_ds = tf.data.Dataset.list_files(valid_path)
valid_ds = valid_ds.map(get_hr_and_lr)
valid_ds = valid_ds.repeat()
valid_ds = valid_ds.batch(1)
