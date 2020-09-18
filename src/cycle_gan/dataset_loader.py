import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.convert_image_dtype(image, tf.float32)

    return image


def image_folder_to_dataset(folder_path, batch_size=64, buffer_size=1000):
    path = pathlib.Path(folder_path)
    files = list(map(lambda a: str(a).encode('utf-8'), path.glob('*.jpg')))

    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(buffer_size).batch(batch_size)

    return dataset
