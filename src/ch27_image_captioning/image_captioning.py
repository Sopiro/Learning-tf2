import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
from ch27_image_captioning.models import *

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

# Download caption annotation files

BASE_PATH = os.path.abspath('../../datasets')

annotation_folder = '/annotations/'

if not os.path.exists(BASE_PATH + annotation_folder):
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                             cache_subdir=os.path.abspath('..'),
                                             origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                             extract=True)
    annotation_train = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'
    annotation_val = os.path.dirname(annotation_zip) + '/annotations/captions_val2014.json'
    os.remove(annotation_zip)
else:
    annotation_train = BASE_PATH + '/annotations/captions_train2014.json'
    annotation_val = BASE_PATH + '/annotations/captions_val2014.json'

# Download image files
image_folder_train = '/train2014/'
if not os.path.exists(BASE_PATH + image_folder_train):
    image_zip = tf.keras.utils.get_file('train2014.zip',
                                        cache_subdir=os.path.abspath('..'),
                                        origin='http://images.cocodataset.org/zips/train2014.zip',
                                        extract=True)
    PATH_COCO_TRAIN = os.path.dirname(image_zip) + image_folder_train
    os.remove(image_zip)
else:
    PATH_COCO_TRAIN = BASE_PATH + image_folder_train

# Download image files
image_folder_val = '/val2014/'
if not os.path.exists(BASE_PATH + image_folder_val):
    image_zip = tf.keras.utils.get_file('val2014.zip',
                                        cache_subdir=os.path.abspath('..'),
                                        origin='http://images.cocodataset.org/zips/val2014.zip',
                                        extract=True)
    PATH_COCO_VAL = os.path.dirname(image_zip) + image_folder_val
    os.remove(image_zip)
else:
    PATH_COCO_VAL = BASE_PATH + image_folder_val

PATH_FLICKR = BASE_PATH + '/flickr30k_images/flickr30k_images/'

# Read the json file
with open(annotation_train, 'r') as f:
    annotations_train = json.load(f)

with open(annotation_val, 'r') as f:
    annotations_val = json.load(f)

# Store captions and image names in vectors
all_captions = []
all_img_name_vector = []

dup = [False] * 600000

# Append COCO train captions
for annot in annotations_train['annotations']:  # ex : {'image_id': 318556, 'id': 48, 'caption': 'A very clean and well decorated empty bathroom'}
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']

    # if dup[image_id]:
    #     continue
    # dup[image_id] = True

    full_image_path = PATH_COCO_TRAIN + 'COCO_train2014_' + '%012d.jpg' % image_id

    all_img_name_vector.append(full_image_path)
    all_captions.append(caption)

# Append COCO val captions
for annot in annotations_val['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']

    # if dup[image_id]:
    #     continue
    # dup[image_id] = True

    full_image_path = PATH_COCO_VAL + 'COCO_val2014_' + '%012d.jpg' % image_id

    all_img_name_vector.append(full_image_path)
    all_captions.append(caption)

# print(len(all_captions), len(all_img_name_vector))  # 123287

flickr_dataset = pd.read_csv(PATH_FLICKR + 'results.csv', delimiter='|')
flickr_dataset = flickr_dataset.to_numpy()

# Append Flickr3k captions
for image_name, comment_number, comment in flickr_dataset:
    if type(comment) != str:
        continue

    # if int(comment_number) != 0:
    #     continue

    caption = '<start> ' + str(comment) + ' <end>'

    full_image_path = PATH_FLICKR + image_name
    all_img_name_vector.append(full_image_path)
    all_captions.append(caption)

# print(len(all_captions), len(all_img_name_vector))  # 123287
# assert False

# Shuffle captions and image_names together
# Set a random state, which always guaranteed to have the same shuffle
train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=1)

# Select the first N captions from the shuffled set
# num_examples = 1024
# train_captions = train_captions[:num_examples]
# img_name_vector = img_name_vector[:num_examples]


print(len(train_captions), len(all_captions))  # 30000 414113


# print(train_captions[:3])


# Function for preprocessing
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


# Initialize Inception-V3 with pretrained weight
image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# Make unique with sorted(set)
encode_train = sorted(set(img_name_vector))

image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

# Disk-caching the features extracted from InceptionV3
# You just have got to do this once
# for img, path in tqdm.tqdm(image_dataset):
#     batch_features = image_features_extract_model(img)
#     batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
#
#     for bf, p in zip(batch_features, path):
#         path_of_feature = p.numpy().decode("utf-8")
#         np.save(path_of_feature, bf.numpy())


# Choose the top 5000 words from the vocabulary
num_words = 20000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
# train_seqs = tokenizer.texts_to_sequences(train_captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# print(tokenizer.index_word[0])  # <pad>
# print(tokenizer.index_word[1])  # <unk>
# print(tokenizer.index_word[2])  # a
# print(tokenizer.index_word[3])  # <start>
# print(tokenizer.index_word[4])  # <end>
# print(tokenizer.index_word[5])  # on
# assert False

# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)

# print(tokenizer.sequences_to_texts(train_seqs[:10]))
# assert False

# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# Calculates the max_length, which is used to store the attention weights
max_length = max(len(t) for t in train_seqs)  # 49

# Create training and validation sets using an 80-20 split
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.1,
                                                                    random_state=0)

# print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))

# print(img_name_train[:5])
# print(cap_train[:5])
# assert False

EPOCHS = 0
REPORT_PER_BATCH = 100
EPOCHS_TO_SAVE = 1
BATCH_SIZE = 100
BUFFER_SIZE = 10000
embedding_dim = 128
feature_dim = 64
rnn_units = 512
fc_units = 1024
vocab_size = num_words + 1
steps_per_epoch = len(img_name_train) // BATCH_SIZE
steps_per_epoch_val = len(img_name_val) // BATCH_SIZE
regularization_rate = 0.001

# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64


# Load the numpy files`
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2:
                      tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Validation dataset
dataset_val = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
dataset_val = dataset_val.map(lambda item1, item2:
                              tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset_val = dataset_val.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE)
dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

encoder = CNN_Encoder(feature_dim)
decoder = RNN_Decoder(embedding_dim, rnn_units, fc_units, vocab_size)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    # real = (batch_size,) = (240,)
    # pred = (batch_size, top_k + 1) = (240, 5001)
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # Mask to skip <pad>
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# Checkpoints
checkpoint_path = "./checkpoints/train9"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1]) * EPOCHS_TO_SAVE
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)


@tf.function
def train_step(img_tensor, target):
    loss = 0

    # print(img_tensor) # shape = (240, 64, 2048) = (batch_size, attention_features_shape, features_shape)
    # print(target) # shape = (240, 49) = (batch_size, max_length)

    # initializing the hidden state for each batch because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])  # shape = (batch_size, units)

    # (batch_size, '<start>'), shape = (batch_size, 1)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        # Encode InceptionV3 features into embedding vector
        features = encoder(img_tensor)  # shape = (batch_size, 64, embedding_dim) = (240, 64, 256)

        # Start from index 1 to jump the start token
        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            # predictions shape = (batch_size, vocab_size)
            predictions, hidden, _ = decoder(dec_input, features, hidden, training=True)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

        # l2 regularization on weights. Not applied on biases.
        trainable_variables = encoder.trainable_variables + decoder.trainable_variables
        loss += tf.reduce_sum([tf.nn.l2_loss(t) for t in trainable_variables if 'bias' not in t.name]) * regularization_rate

    avg_loss = (loss / int(target.shape[1]))

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return avg_loss


@tf.function
def calc_validation_loss(img_tensor, target):
    loss = 0

    # print(img_tensor) # shape = (240, 64, 2048) = (batch_size, attention_features_shape, features_shape)
    # print(target) # shape = (240, 49) = (batch_size, max_length)

    # initializing the hidden state for each batch because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])  # shape = (batch_size, units)

    # (batch_size, '<start>'), shape = (batch_size, 1)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    # Encode InceptionV3 features into embedding vector
    features = encoder(img_tensor)  # shape = (batch_size, 64, embedding_dim) = (240, 64, 256)

    # Start from index 1 to jump the start token
    for i in range(1, target.shape[1]):
        # passing the features through the decoder
        # predictions shape = (batch_size, vocab_size)
        predictions, hidden, _ = decoder(dec_input, features, hidden)

        loss += loss_function(target[:, i], predictions)

        # using teacher forcing
        dec_input = tf.expand_dims(target[:, i], 1)

    avg_loss = (loss / int(target.shape[1]))

    return avg_loss


print('Start Epoch = ', start_epoch)
print('Start training for {} epochs'.format(EPOCHS))
print('Batch Size = ', BATCH_SIZE)
print('Steps per epoch = ', steps_per_epoch)

loss_train_plot = []
loss_val_plot = []
loss_train_file = checkpoint_path + '/loss_train.npy'
loss_val_file = checkpoint_path + '/loss_val.npy'

if os.path.exists(loss_train_file):
    loss_train_plot = np.load(loss_train_file)

if os.path.exists(loss_val_file):
    loss_val_plot = np.load(loss_val_file)

for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0

    current_epoch = start_epoch + epoch + 1

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss = train_step(img_tensor, target)
        total_loss += batch_loss

        if batch % REPORT_PER_BATCH == 0:
            print('Epoch {} Batch {}/{} Loss {:.4f}'.format(current_epoch, batch, steps_per_epoch, batch_loss))
            loss_train_plot = np.append(loss_train_plot, batch_loss)

    total_loss_val = 0

    # Print out validation loss
    for (itr, (img_tensor_val, target_val)) in enumerate(dataset_val):
        batch_loss_val = calc_validation_loss(img_tensor_val, target_val)
        total_loss_val += batch_loss_val

    print('Epoch {} Loss {:.6f}'.format(current_epoch, total_loss / steps_per_epoch))
    loss_train_plot = np.append(loss_train_plot, total_loss / steps_per_epoch)
    print('Epoch {} Validation Loss {:.6f}'.format(current_epoch, total_loss_val / steps_per_epoch_val))
    loss_val_plot = np.append(loss_val_plot, total_loss_val / steps_per_epoch_val)

    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    if (epoch + 1) % EPOCHS_TO_SAVE == 0:
        ckpt_manager.save()

val_gap = 2 + steps_per_epoch // REPORT_PER_BATCH

np.save(loss_train_file, loss_train_plot)
np.save(loss_val_file, loss_val_plot)

loss_val_plot = np.r_[loss_val_plot[0], loss_val_plot]

plt.plot(loss_train_plot, 'b', label='train loss')
plt.plot(np.arange(len(loss_val_plot)) * val_gap, loss_val_plot, 'r--', label='val loss')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()


def evaluate(image):
    attention_plot = np.zeros(shape=(max_length, attention_features_shape))

    temp_input = tf.expand_dims(load_image(image)[0], 0)  # Expand batch axis
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)  # shape = (1, 64, embedding_dim)

    hidden = decoder.reset_state(batch_size=1)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        # predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


# captions on the validation set
for it in range(0):
    rid = np.random.randint(0, len(img_name_val))
    image = img_name_val[rid]
    real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
    result, attention_plot = evaluate(image)

    print('Real Caption:', real_caption)
    print('Prediction Caption:', ' '.join(result))
    plot_attention(image, result, attention_plot)

# assert False

# image_url = 'https://tensorflow.org/images/surf.jpg'
# image_url = 'https://upload.wikimedia.org/wikipedia/commons/4/45/A_small_cup_of_coffee.JPG'
# image_url = 'https://post-phinf.pstatic.net/MjAxOTAyMTVfMjc2/MDAxNTUwMjA4NzE2MTIy.-Cae85qV570pF0FsWyoF2P4oEdooap7xS5vyfr3cGXUg.UaJFjECmhav26t5L985R9eg_cVS8zEDmyj_ihBrPR3wg.JPEG/3.jpg?type=w1200'
# image_url = 'https://raw.githubusercontent.com/yashk2810/Image-Captioning/master/images/frisbee.png'
image_url = 'https://pds.joins.com/news/component/htmlphoto_mmdata/201910/30/f06c4fe8-dfa9-4ae4-af32-1aa0505d5cb1.jpg'
image_extension = image_url[-4:]
full_image_path = tf.keras.utils.get_file('pink_dressed_woman' + image_extension, origin=image_url)

result, attention_plot = evaluate(full_image_path)
print('Prediction Caption:', ' '.join(result))
plot_attention(full_image_path, result, attention_plot)
# opening the image
Image.open(full_image_path)
