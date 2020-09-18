import tensorflow as tf
import matplotlib.pyplot as plt
from model import *
import pathlib
import tensorflow_datasets as tfds
from dataset_loader import *
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

BUFFER_SIZE = 1000
BATCH_SIZE = 16
IMG_WIDTH = 256
IMG_HEIGHT = 256

EPOCHS = 40
LAMBDA = 10

base = os.path.abspath('.')

train_A = image_folder_to_dataset('dataset/trainA', batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)
train_B = image_folder_to_dataset('dataset/trainB', batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)
test_A = image_folder_to_dataset('dataset/testA', batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)
test_B = image_folder_to_dataset('dataset/testB', batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)

# print(train_A.take(1))
# plt.figure(figsize=(10, 10))
# for images in train_A.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy())
#         plt.axis("off")
#
# plt.show()

train_A = train_A.unbatch()
train_B = train_B.unbatch()
test_A = test_A.unbatch()
test_B = test_B.unbatch()


def random_crop(image):
    cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image


# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)

    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image


def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image):
    image = normalize(image)
    return image


train_A = train_A.map(preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
train_B = train_B.map(preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
test_A = test_A.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
test_B = test_B.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)

sample_A = next(iter(train_A))
sample_B = next(iter(train_B))

# plt.subplot(121)
# plt.title('Horse')
# plt.imshow(sample_A[0] * 0.5 + 0.5)
#
# plt.subplot(122)
# plt.title('Horse with random jitter')
# plt.imshow(random_jitter(sample_A[0]) * 0.5 + 0.5)
#
# plt.show()

OUTPUT_CHANNELS = 3

generator_a2b = Generator(OUTPUT_CHANNELS)
generator_b2a = Generator(OUTPUT_CHANNELS)

discriminator_a = Discriminator()
discriminator_b = Discriminator()

to_zebra = generator_a2b(sample_A)
to_horse = generator_b2a(sample_B)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_A, to_zebra, sample_B, to_horse]
title = ['A', 'To B', 'B', 'To A']

# for i in range(len(imgs)):
#     plt.subplot(2, 2, i + 1)
#     plt.title(title[i])
#     if i % 2 == 0:
#         plt.imshow(imgs[i][0] * 0.5 + 0.5)
#     else:
#         plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
# plt.show()
#
# plt.figure(figsize=(8, 8))

# plt.subplot(121)
# plt.title('Is a real A?')
# plt.imshow(discriminator_a(sample_A)[0, ..., -1], cmap='RdBu_r')
#
# plt.subplot(122)
# plt.title('Is a real B?')
# plt.imshow(discriminator_b(sample_B)[0, ..., -1], cmap='RdBu_r')
#
# plt.show()


loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def cycle_loss(real, cycled):
    loss = tf.reduce_mean(tf.abs(real - cycled))

    return loss


def identity_loss(real, same):
    loss = tf.reduce_mean(tf.abs(real - same))

    return loss


generator_a2b_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_b2a_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_a_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_b_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_path = "./checkpoints/train1"

ckpt = tf.train.Checkpoint(generator_a2b=generator_a2b,
                           generator_b2a=generator_b2a,
                           discriminator_a=discriminator_a,
                           discriminator_b=discriminator_b,
                           generator_a2b_optimizer=generator_a2b_optimizer,
                           generator_b2a_optimizer=generator_b2a_optimizer,
                           discriminator_a_optimizer=discriminator_a_optimizer,
                           discriminator_b_optimizer=discriminator_b_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')
