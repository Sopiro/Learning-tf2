import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

style_path = tf.keras.utils.get_file('style.jpg', 'http://bit.ly/2mGfZIq')

style_image = plt.imread(style_path)
style_image = cv2.resize(style_image, dsize=(224, 224))
style_image = style_image / 255.0

plt.imshow(style_image)
plt.show()

target_image = tf.random.uniform(style_image.shape)
# print(target_image[0, 0, :])
# plt.imshow(target_image)
# plt.show()

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# for layer in vgg.layers:
#     print(layer.name)

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

vgg.trainable = False
outputs = [vgg.get_layer(name).output for name in style_layers]
model = tf.keras.Model(inputs=[vgg.input], outputs=outputs)


def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)

    return gram / tf.cast(n, tf.float32)


style_batch = style_image.astype('float32')
style_batch = tf.expand_dims(style_batch, axis=0)
style_output = model.predict(tf.keras.applications.vgg19.preprocess_input(style_batch * 255.0))

# print(style_output[0].shape)
# plt.imshow(tf.squeeze(style_output[0][:, :, :, 0], 0), cmap='gray')
# plt.show()

style_outputs = [gram_matrix(out) for out in style_output]

plt.figure(figsize=(12, 10))
for c in range(5):
    plt.subplot(3, 2, c + 1)
    array = sorted(style_outputs[c].numpy()[0].tolist())
    array = array[::-1]
    plt.bar(range(style_outputs[c].shape[0]), array)
    plt.title(style_layers[c])
plt.show()
