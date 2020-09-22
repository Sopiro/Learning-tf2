import time
import matplotlib.pyplot as plt
from model import *
from dataset_loader import *
import tqdm

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

REAL_LABEL = 1.0
BUFFER_SIZE = 5000
IMG_WIDTH = 256
IMG_HEIGHT = 256

APPEND_IDENTITY_LOSS = True
USE_LSGAN = True

version = 1

if version == 1:
    dsdir = 'monet2photo'
    checkpoint_path = "./checkpoints/monet"
elif version == 2:
    dsdir = 'vangogh2photo'
    checkpoint_path = "./checkpoints/gogh"
elif version == 3:
    dsdir = 'face2ckpt'
    checkpoint_path = "./checkpoints/ckpt"
elif version == 4:
    dsdir = 'horse2zebra'
    checkpoint_path = "./checkpoints/zebra"
else:
    assert False

train_A = image_folder_to_dataset('dataset/{}/trainA'.format(dsdir), buffer_size=BUFFER_SIZE)
train_B = image_folder_to_dataset('dataset/{}/trainB'.format(dsdir), buffer_size=BUFFER_SIZE)
test_A = image_folder_to_dataset('dataset/{}/testA'.format(dsdir), buffer_size=BUFFER_SIZE)
test_B = image_folder_to_dataset('dataset/{}/testB'.format(dsdir), buffer_size=BUFFER_SIZE)
custom = image_folder_to_dataset('dataset/custom', buffer_size=100)

print('Domain A images :', len(train_A))
print('Domain B images :', len(train_B))


# print(train_A.take(1))
# plt.figure(figsize=(10, 10))
# for images in train_A.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy())
#         plt.axis("off")
#
# plt.show()

def resize(image):
    resized_image = tf.image.resize(image, size=(IMG_WIDTH, IMG_HEIGHT))
    return resized_image


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
    # image = resize(image)
    image = normalize(image)
    return image


train_A = train_A.map(preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE).cache()
train_B = train_B.map(preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE).cache()
test_A = test_A.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE).cache()
test_B = test_B.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE).cache()
custom = custom.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE).cache()

sample_A = next(iter(train_A.batch(1)))
sample_B = next(iter(train_B.batch(1)))
sample_C = next(iter(custom.batch(1).shuffle(1, 5)))

# plt.subplot(121)
# plt.title('Horse')
# plt.imshow(sample_A[0] * 0.5 + 0.5)
#
# plt.subplot(122)
# plt.title('Horse with random jitter')
# plt.imshow(random_jitter(sample_A[0]) * 0.5 + 0.5)
#
# plt.show()

generator_a2b = ResNetGenerator()
generator_b2a = ResNetGenerator()
discriminator_a = Discriminator()
discriminator_b = Discriminator()

to_zebra = generator_a2b(sample_A)
to_horse = generator_b2a(sample_B)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_A, to_zebra, sample_B, to_horse]
title = ['A', 'To B', 'B', 'To A']

for i in range(len(imgs)):
    plt.subplot(2, 2, i + 1)
    plt.title(title[i])
    if i % 2 == 0:
        plt.imshow(imgs[i][0] * 0.5 + 0.5)
    else:
        plt.imshow(imgs[i][0] * 0.5 * 8 + 0.5)
plt.show()

plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real A?')
plt.imshow(discriminator_a(sample_A)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real B?')
plt.imshow(discriminator_b(sample_B)[0, ..., -1], cmap='RdBu_r')

plt.show()

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, generated, use_lsgan=True):
    if use_lsgan:
        real_loss = tf.reduce_mean(tf.math.squared_difference(real, REAL_LABEL))
        generated_loss = tf.reduce_mean(tf.math.square(generated))
    else:
        real_loss = loss_obj(tf.ones_like(real), real)
        generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5


def generator_loss(generated, use_lsgan=True):
    if use_lsgan:
        loss = tf.reduce_mean(tf.math.squared_difference(generated, REAL_LABEL))
    else:
        loss = loss_obj(tf.ones_like(generated), generated)

    return loss


def cycle_consistency_loss(real, cycled):
    loss = tf.reduce_mean(tf.abs(real - cycled))
    return loss


def identity_loss(real, same):
    loss = tf.reduce_mean(tf.abs(real - same))
    return loss


generator_a2b_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_b2a_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_a_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_b_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

ckpt = tf.train.Checkpoint(generator_a2b=generator_a2b,
                           generator_b2a=generator_b2a,
                           discriminator_a=discriminator_a,
                           discriminator_b=discriminator_b,
                           generator_a2b_optimizer=generator_a2b_optimizer,
                           generator_b2a_optimizer=generator_b2a_optimizer,
                           discriminator_a_optimizer=discriminator_a_optimizer,
                           discriminator_b_optimizer=discriminator_b_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)


def generate_images(model, test_input):
    prediction = model(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


BATCH_SIZE = 2
EPOCHS = 3
LAMBDA = 10
EPOCHS_TO_SAVE = 1
REPORT_PER_BATCH = 10
STEPS_PER_EPOCH = min(len(train_A), len(train_B)) // BATCH_SIZE

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1]) * EPOCHS_TO_SAVE
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint Epoch {} restored'.format(start_epoch))

train_A = train_A.batch(BATCH_SIZE)
train_B = train_B.batch(BATCH_SIZE)
test_A = test_A.batch(BATCH_SIZE)
test_B = test_B.batch(BATCH_SIZE)

print('Start Epoch = ', start_epoch)
print('Start training for {} epochs'.format(EPOCHS))
print('Batch Size = ', BATCH_SIZE)
print('Steps per epoch = ', STEPS_PER_EPOCH)

# Run the trained model on the test dataset
for inp in test_A.take(5):
    generate_images(generator_a2b, inp)

generate_images(generator_b2a, sample_C)


@tf.function
def train_step(real_a, real_b):
    with tf.GradientTape(persistent=True) as tape:
        fake_b = generator_a2b(real_a, training=True)
        cycled_a = generator_b2a(fake_b, training=True)

        fake_a = generator_b2a(real_b, training=True)
        cycled_b = generator_a2b(fake_a, training=True)

        same_a = generator_b2a(real_a, training=True)
        same_b = generator_a2b(real_b, training=True)

        disc_real_a = discriminator_a(real_a, training=True)
        disc_real_b = discriminator_b(real_b, training=True)

        disc_fake_a = discriminator_a(fake_a, training=True)
        disc_fake_b = discriminator_b(fake_b, training=True)

        # calculate the loss
        gen_a2b_loss = generator_loss(disc_fake_b, use_lsgan=USE_LSGAN)
        gen_b2a_loss = generator_loss(disc_fake_a, use_lsgan=USE_LSGAN)

        total_cycle_consistency_loss = (cycle_consistency_loss(real_a, cycled_a) + cycle_consistency_loss(real_b, cycled_b)) * LAMBDA

        # Total generator loss = adversarial loss + cycle loss
        total_gen_a2b_loss = gen_a2b_loss + total_cycle_consistency_loss
        total_gen_b2a_loss = gen_b2a_loss + total_cycle_consistency_loss

        if APPEND_IDENTITY_LOSS:
            total_gen_a2b_loss += 0.5 * LAMBDA * identity_loss(real_b, same_b)
            total_gen_b2a_loss += 0.5 * LAMBDA * identity_loss(real_a, same_a)

        disc_a_loss = discriminator_loss(disc_real_a, disc_fake_a, use_lsgan=USE_LSGAN)
        disc_b_loss = discriminator_loss(disc_real_b, disc_fake_b, use_lsgan=USE_LSGAN)

    # Calculate the gradients for generator and discriminator
    generator_a2b_gradients = tape.gradient(total_gen_a2b_loss, generator_a2b.trainable_variables)
    generator_b2a_gradients = tape.gradient(total_gen_b2a_loss, generator_b2a.trainable_variables)

    discriminator_a_gradients = tape.gradient(disc_a_loss, discriminator_a.trainable_variables)
    discriminator_b_gradients = tape.gradient(disc_b_loss, discriminator_b.trainable_variables)

    # Apply the gradients to the optimizer
    generator_a2b_optimizer.apply_gradients(zip(generator_a2b_gradients, generator_a2b.trainable_variables))
    generator_b2a_optimizer.apply_gradients(zip(generator_b2a_gradients, generator_b2a.trainable_variables))
    discriminator_a_optimizer.apply_gradients(zip(discriminator_a_gradients, discriminator_a.trainable_variables))
    discriminator_b_optimizer.apply_gradients(zip(discriminator_b_gradients, discriminator_b.trainable_variables))


for epoch in range(EPOCHS):
    start = time.time()

    current_epoch = start_epoch + epoch + 1

    # n = 0
    print('-------------------------------------------------------------')
    for image_a, image_b in tqdm.tqdm(tf.data.Dataset.zip((train_A, train_B))):
        train_step(image_a, image_b)
        # if n % REPORT_PER_BATCH == 0:
        #     print('Epoch {} Batch {}/{}'.format(current_epoch, n, STEPS_PER_EPOCH))
        # n += 1

    if (epoch + 1) % EPOCHS_TO_SAVE == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(current_epoch, ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(current_epoch, time.time() - start))

    # Using a consistent image (sample_A) so that the progress of the model is clearly visible.
    # generate_images(generator_b2a, sample_C)

# Run the trained model on the test dataset
for inp in test_B.take(5):
    generate_images(generator_b2a, inp)

generate_images(generator_b2a, sample_C)
