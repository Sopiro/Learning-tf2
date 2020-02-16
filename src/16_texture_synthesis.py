import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import IPython.display as display
import time
import imageio

# Download image from URL and store it to keras's cache folder (C:\Users\(UserName)\.keras\datasets)
style_image_path = tf.keras.utils.get_file('style.jpg', 'http://bit.ly/2mGfZIq')

style_image = plt.imread(style_image_path)
style_image = cv2.resize(style_image, dsize=(256, 256))
style_image = style_image / 255.0

# Initialize target image with random noise
target_image = tf.random.uniform(shape=(256, 256, 3))

# Import VGG-19 network where the weights fit to imagenet
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# Check out the layers
# for layer in vgg.layers:
#     print(layer.name)

conv_layers = ['block1_conv1',
               'block2_conv1',
               'block3_conv1',
               'block4_conv1',
               'block5_conv1']

# Extract convolution layer outputs and set to model's output
vgg.trainable = False
conv_outs = [vgg.get_layer(name).output for name in conv_layers]
model = tf.keras.Model(inputs=[vgg.input], outputs=conv_outs)


# Calculate gram matrix of Convolution outputs
def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    size = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(size, tf.float32)


# Process forward to network
style_batch = style_image.astype('float32')
style_batch = tf.expand_dims(style_batch, axis=0)  # Insert dimension for batching
style_output = model.call(tf.keras.applications.vgg19.preprocess_input(style_batch * 255.0))

# Visualize convolution layer output
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


# Calculate and get gram matrices of target image
def get_outputs(image):
    image_batch = tf.expand_dims(image, axis=0)
    output = model.call(image_batch * 255.0)
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

total_variation_weight = 1e10
style_weight = 1e-3


# Use tensorflow's autograph function.
# Let tensorflow generate graph from this function
# tape.gradient(loss, variables) traces all calculation occurred in with syntax.
# Computes the gradient using operations recorded in context of this tape.
# Returns ∂loss/∂variables.
@tf.function
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = get_outputs(image)
        loss = style_weight * get_loss(outputs, style_outputs) + total_variation_weight * total_variation_loss(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients(grads_and_vars=[(grad, image)])
    image.assign(clip_0_1(image))


start = time.time()

image = tf.Variable(target_image)

epochs = 100
step_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(step_per_epoch):
        step += 1
        train_step(image)

    if n % 10 == 0:
        imageio.imwrite('../systhesis_result/epoch_{0}.png'.format(n), image.read_value().numpy())

imageio.imwrite('../systhesis_result/result.png', image.read_value().numpy())

display.clear_output(wait=True)
plt.imshow(image.read_value())
plt.show()

end = time.time()

print("Total time passed: {:.1f}".format(end - start))
