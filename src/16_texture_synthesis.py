import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import IPython.display as display
import time
import imageio

style_path = tf.keras.utils.get_file('style.jpg', 'http://bit.ly/2mGfZIq')
# style_path = tf.keras.utils.get_file('style2.jpg', '')

style_image = plt.imread(style_path)
style_image = cv2.resize(style_image, dsize=(224, 224))
style_image = style_image / 255.0

# plt.imshow(style_image)
# plt.show()

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
conv_outs = [vgg.get_layer(name).output for name in style_layers]
model = tf.keras.Model(inputs=[vgg.input], outputs=conv_outs)


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


# plt.figure(figsize=(12, 10))
# for c in range(5):
#     plt.subplot(3, 2, c + 1)
#     array = sorted(style_outputs[c].numpy()[0].tolist())
#     array = array[::-1]
#     plt.bar(range(style_outputs[c].shape[0]), array)
#     plt.title(style_layers[c])
# plt.show()


def get_outputs(image):
    image_batch = tf.expand_dims(image, axis=0)
    output = model(image_batch * 255.0)
    outputs = [gram_matrix(out) for out in output]

    return outputs


# Calculate MSE loss between target texture and style texture
def get_loss(outputs, style_outputs):
    return tf.reduce_sum([tf.reduce_mean((o - s) ** 2) for o, s in zip(outputs, style_outputs)])


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def high_pass_x_y(image):
    x_var = image[:, 1:, :] - image[:, :-1, :]
    y_var = image[1:, :, :] - image[:-1, :, :]
    return x_var, y_var


def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(x_deltas ** 2) + tf.reduce_mean(y_deltas ** 2)


opt = tf.keras.optimizers.Adam(learning_rate=0.2, beta_1=0.99, epsilon=1e-1)

total_variation_weight = 1e8
style_weight = 1e-2


@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = get_outputs(image)
        loss = style_weight * get_loss(outputs, style_outputs) + total_variation_weight * total_variation_loss(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


start = time.time()

image = tf.Variable(target_image)

epochs = 50
step_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(step_per_epoch):
        step += 1
        train_step(image)

    imageio.imwrite('../systhesis_result/epoch_{0}.png'.format(n), image.read_value().numpy())

display.clear_output(wait=True)
plt.imshow(image.read_value())
plt.show()

end = time.time()

print("Total time: {:.1f}".format(end - start))
