import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time

(train_images, _), _ = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256
assert BUFFER_SIZE <= train_images.__sizeof__()

train_dataset = tf.data.Dataset.from_tensor_slices(train_images[:BUFFER_SIZE]).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = tf.keras.Sequential([
    tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Reshape((7, 7, 256)),

    tf.keras.layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Conv2DTranspose(1, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False),
    tf.keras.layers.Activation('tanh')
])

# generator.summary()
# assert False

# noise = tf.random.normal((1, 100))
# generated_image = generator(noise, training=False)
#
# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()

discriminator = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

# Binary_Cross_Entropy(y, h(x)) = -(y * log(h(x)) + (1 - y) * log(1 - h(x)))
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# print(tf.ones_like(tf.random.normal((10, 10))))


# Recognize real as real, fake as fake
def loss_d(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


# Recognize fake as real
def loss_g(fake_output):
    fake_loss = bce(tf.ones_like(fake_output), fake_output)
    return fake_loss


optimizer_g = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999)
optimizer_d = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999)

EPOCHS = 30

checkpoint_dir = './gan_training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer_g=optimizer_g,
                                 optimizer_d=optimizer_d,
                                 generator=generator,
                                 discriminator=discriminator,
                                 epochs=tf.Variable(0))
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_image, training=True)

        gen_loss = loss_g(fake_output)
        disc_loss = loss_d(real_output, fake_output)

    gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disk_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    optimizer_g.apply_gradients(zip(gen_grad, generator.trainable_variables))
    optimizer_d.apply_gradients(zip(disk_grad, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        checkpoint.epochs.assign_add(1)

        for image_batch in dataset:
            train_step(image_batch)

        generate_and_save_images(generator, epoch + 1, seed)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(checkpoint.epochs.numpy(), time.time() - start))


def generate_and_save_images(model, epoch, test_input, show=False):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(8, 8))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
    plt.savefig('./gan_result/image_at_epoch_{:04d}.png'.format(checkpoint.epochs.numpy()))
    if show:
        plt.show()
    plt.close(fig)


train(train_dataset, EPOCHS)

print(discriminator(tf.expand_dims(train_images[0], 0)))
print(discriminator(generator(tf.random.normal([1, noise_dim]))))

for i in range(1):
    generate_and_save_images(generator, 1000 + i, tf.random.normal([num_examples_to_generate, noise_dim]), True)
