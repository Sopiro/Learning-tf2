import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2

labels = tf.keras.utils.get_file('imagenet_labels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
img = tf.keras.utils.get_file('ipod.jpg', 'https://i.etsystatic.com/19007236/r/il/dd3a32/2133570822/il_570xN.2133570822_qkhj.jpg')

url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'

m = tf.keras.Sequential([
    hub.KerasLayer(handle=url, output_shape=[1001])
])

label_text = None

with open(labels, 'r') as f:
    label_text = f.read().split('\n')[:-1]

label_text = np.array(label_text)
# print(label_text[:10])


m.build([None, 224, 224, 3])  # Batch input shape.

img = cv2.imread(img)
img = cv2.resize(img, dsize=(224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

predicted = m.predict(img)[0]

top_5_predicted = predicted.argsort()[::-1][:5]

print(label_text[top_5_predicted])
print(predicted[top_5_predicted])