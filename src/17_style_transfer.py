import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import IPython.display as display
import time
import imageio

preserve_color = False

# Download images from URL and store it to keras's cache folder (C:\Users\(UserName)\.keras\datasets)
# style_image_path = tf.keras.utils.get_file('style.jpg', 'http://bit.ly/2mGfZIq')
# content_path = tf.keras.utils.get_file('content.jpg', 'http://bit.ly/2mAfUX1')
style_image_path = tf.keras.utils.get_file('oil.jpg', '')
content_path = tf.keras.utils.get_file('mina.jpg', '')

# Pre-Processing for style image
style_image = plt.imread(style_image_path)
style_image = cv2.resize(style_image, dsize=(1024, 1024))
style_image = style_image / 255.0

# Pre-Processing for content image
content_image = plt.imread(content_path)
max_dim = 512
long_dim = max(content_image.shape[:-1])
scale = max_dim / long_dim
new_height = int(content_image.shape[0] * scale)
new_width = int(content_image.shape[1] * scale)
content_image = cv2.resize(content_image, dsize=(new_width, new_height))
content_image = content_image / 255.0
# plt.imshow(content_image)
# plt.show()

# Initialize target image with random noise
target_image = tf.random.uniform(shape=(new_height, new_width, 3))
# target_image = tf.image.grayscale_to_rgb(target_image)
# target_image = tf.image.resize(target_image, (512, 512))

# plt.imshow(target_image)
# plt.show()
# assert False

# Import VGG-19 network where the weights fit to imagenet
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# Check out the layers
for layer in vgg.layers:
    print(layer.name)

conv_layers_for_style = ['block1_conv1',
                         'block1_conv2',
                         'block2_conv1',
                         'block2_conv2',
                         'block3_conv1',
                         'block3_conv2',
                         'block3_conv3',
                         'block3_conv4',
                         'block4_conv1',
                         'block4_conv2',
                         'block4_conv3',
                         'block4_conv4',
                         'block5_conv1',
                         'block5_conv2',
                         'block5_conv3',
                         'block5_conv4']
conv_layers_for_content = ['block5_conv2']

# Extract convolution layer outputs and set to model's output
vgg.trainable = False
conv_outs_for_style = [vgg.get_layer(name).output for name in conv_layers_for_style]
model_for_style = tf.keras.Model(inputs=[vgg.input], outputs=conv_outs_for_style)

conv_outs_for_content = [vgg.get_layer(name).output for name in conv_layers_for_content]
model_for_content = tf.keras.Model(inputs=[vgg.input], outputs=conv_outs_for_content)


# Calculate gram matrix of Convolution outputs
def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    size = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(size, tf.float32)


# Process forward to network
style_batch = style_image.astype('float32')
if preserve_color:
    style_batch = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(style_batch))
style_batch = tf.expand_dims(style_batch, axis=0)  # Insert dimension for batching
style_output = model_for_style(tf.keras.applications.vgg19.preprocess_input(style_batch * 255.0))

content_batch = content_image.astype('float32')
content_batch = tf.expand_dims(content_batch, axis=0)
content_output = model_for_content(tf.keras.applications.vgg19.preprocess_input(content_batch * 255.0))

# Visualize convolution layer output
# print(style_output[0].shape)
# plt.imshow(tf.squeeze(style_output[0][:, :, :, 0], 0), cmap='gray')
# plt.show()

style_gram_matrices = [gram_matrix(out) for out in style_output]


# plt.figure(figsize=(12, 10))
# for c in range(5):
#     plt.subplot(3, 2, c + 1)
#     array = sorted(style_outputs[c].numpy()[0].tolist())
#     array = array[::-1]
#     plt.bar(range(style_outputs[c].shape[0]), array)
#     plt.title(style_layers[c])
# plt.show()


# Calculate and get gram matrices of target image
def get_style_outputs(image):
    image_batch = tf.expand_dims(image, axis=0)
    output = model_for_style(tf.keras.applications.vgg19.preprocess_input(image_batch * 255.0))
    outputs = [gram_matrix(out) for out in output]
    return outputs


# Calculate MSE loss about gram between target texture and style texture
def get_style_loss(outputs, style_outputs):
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


def get_content_output(image):
    image_batch = tf.expand_dims(image, axis=0)
    output = model_for_content(tf.keras.applications.vgg19.preprocess_input(image_batch * 255.0))
    return output


# Calculate MSE loss about pixel data between target texture and style texture
def get_content_loss(image, content_output):
    return tf.reduce_sum(tf.reduce_mean(image - content_output) ** 2)


opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.99, epsilon=1e-1)

total_variation_weight = 2e9
style_weight = 2e-2
content_weight = 2e8


# Use tensorflow's autograph function.
# Let tensorflow generate graph from this function
# tape.gradient(loss, variables) traces all calculation occurred in with syntax.
# Computes the gradient using operations recorded in context of this tape.
# Returns ∂loss/∂variables.
@tf.function
def train_step(image):
    with tf.GradientTape() as tape:
        style_gram_outputs = get_style_outputs(image)
        content_conv_output = get_content_output(image)

        loss = style_weight * get_style_loss(style_gram_outputs, style_gram_matrices)
        loss += total_variation_weight * total_variation_loss(image)
        loss += content_weight * get_content_loss(content_conv_output, content_output)

    grad = tape.gradient(loss, image)
    opt.apply_gradients(grads_and_vars=[(grad, image)])
    image.assign(clip_0_1(image))


start = time.time()

# You can choose to start from random noise or content image.
image = tf.Variable(content_image.astype('float32'))
# image = tf.Variable(target_image.astype('float32'))

epochs = 50
step_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(step_per_epoch):
        step += 1
        train_step(image)

    if n % 10 == 0:
        imageio.imwrite('../transfer_result/epoch_{0}.png'.format(n), image.read_value().numpy())

imageio.imwrite('../transfer_result/result.png', image.read_value().numpy())

display.clear_output(wait=True)
plt.imshow(image.read_value())
plt.title("Total Train steps : {}".format(step))
plt.show()

end = time.time()

print("Total time passed: {:.1f}".format(end - start))
