import tensorflow_hub as hub
import tensorflow as tf
import time


# Function for preprocessing
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (640, 640))

    return tf.expand_dims(tf.cast(img, tf.uint8), 0)


# Function for preprocessing
def load_image2(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, dtype=tf.float32) / 255.0

    return tf.expand_dims(tf.cast(img, tf.float32), 0)


detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1")

img = load_image('C:/Users/Sopiro/Desktop/20200825/dnteo.png')

detector_output = detector(img)
scores = detector_output['detection_scores']
class_ids = detector_output['detection_classes'][:, :64]
detection_boxes = detector_output['detection_boxes'][:, :64]

print(detection_boxes)

res = tf.concat([tf.keras.layers.Embedding(100, 3)(class_ids), detection_boxes], axis=-1)

print(res)
