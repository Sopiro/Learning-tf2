from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import os
import json
import time
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
from ch30_transformer_captioning.models import *
import io
import tensorflow_hub as hub

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

dup = [0] * 600000

# Append COCO train captions
for annot in annotations_train['annotations']:  # ex : {'image_id': 318556, 'id': 48, 'caption': 'A very clean and well decorated empty bathroom'}
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']

    if dup[image_id] == 2:
        continue
    dup[image_id] += 1

    full_image_path = PATH_COCO_TRAIN + 'COCO_train2014_' + '%012d.jpg' % image_id

    all_img_name_vector.append(full_image_path)
    all_captions.append(caption)

# Append COCO val captions
for annot in annotations_val['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']

    if dup[image_id] == 2:
        continue
    dup[image_id] += 1

    full_image_path = PATH_COCO_VAL + 'COCO_val2014_' + '%012d.jpg' % image_id

    all_img_name_vector.append(full_image_path)
    all_captions.append(caption)

# print(len(all_captions), len(all_img_name_vector))  # 123287

flickr_dataset = pd.read_csv(PATH_FLICKR + 'results.csv', delimiter='|')
flickr_dataset = flickr_dataset.to_numpy()

# Append Flickr3k captions
# for image_name, comment_number, comment in flickr_dataset:
#     if type(comment) != str:
#         continue
#
#     if len(str(comment).split(';')) > 1:
#         print(str(comment).split(';')[0])
#
#     if int(comment_number) != 0:
#         continue
#
#     caption = '<start> ' + str(comment) + ' <end>'
#
#     full_image_path = PATH_FLICKR + image_name
#     all_img_name_vector.append(full_image_path)
#     all_captions.append(caption)


# print(len(all_captions), len(all_img_name_vector))  # 123287
# assert False

# Shuffle captions and image_names together
# Set a random state, which always guaranteed to have the same shuffle
train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=1)

# Select the first N captions from the shuffled set
# num_examples = 47
# train_captions = train_captions[:num_examples]
# img_name_vector = img_name_vector[:num_examples]

print('All captions :', len(all_captions))  # 30000 414113


# Function for preprocessing
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img1 = tf.image.resize(img, (640, 640))  # For Faster-RCNN
    img2 = tf.image.resize(img, (299, 299))  # For Efficient-Net

    return tf.cast(img1, tf.uint8), tf.keras.applications.inception_v3.preprocess_input(img2), image_path


# Initialize Inception-V3 with pretrained weight
image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
imagenet_feature_dim_a = 64
imagenet_feature_dim_b = 2048

# print(image_features_extract_model(tf.expand_dims(load_image('C:/Users/Sopiro/Desktop/20200825/irene.png')[0], 0)).shape)
# assert False

# Make unique with sorted(set)
encode_train = sorted(set(img_name_vector))

image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1)

frcnn = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1")

num_detections = 64
frcnn_embedding_dim = 508

assert imagenet_feature_dim_a == num_detections

embedding_table = tf.random.Generator.from_seed(1).uniform((frcnn_embedding_dim, frcnn_embedding_dim))
# Save embedding table
path_embedding_table = './checkpoints/embedding/embedding_table.txt'
# np.savetxt(path_embedding_table, embedding_table.numpy())
loaded = np.loadtxt(path_embedding_table)

assert np.array_equal(embedding_table, loaded)


def extract_feature(img1, img2):
    frcnn_output = frcnn(img1)

    class_ids = tf.cast(frcnn_output['detection_classes'][:, :num_detections], tf.int32)  # (1, 64)
    detection_boxes = frcnn_output['detection_boxes'][:, :num_detections]  # (1, 64, 4)

    frcnn_features = tf.concat([tf.nn.embedding_lookup(embedding_table, class_ids), detection_boxes], axis=-1)  # (1, 64, 508) + (1, 64, 4) -> (1, 64, 512)

    imagenet_features = image_features_extract_model(img2)
    imagenet_features = tf.reshape(imagenet_features, (imagenet_features.shape[0], -1, imagenet_features.shape[3]))  # (1, 64, 2048)

    concatenated_feature = tf.concat([imagenet_features, frcnn_features], axis=-1)  # (1, 64, 2048) + (1, 64, 512) -> (1, 64, 2560)

    return concatenated_feature


# Disk-caching the features extracted from pre-trained model
# You just have got to do this once
# for img1, img2, path in tqdm(image_dataset):
#
#     extracted_feature = extract_feature(img1, img2)
#
#     for bf, p in zip(extracted_feature, path):
#         path_of_feature = p.numpy().decode("utf-8")
#         np.save(path_of_feature, bf.numpy())
#
# assert False

# Choose the top 5000 words from the vocabulary
num_words = 20000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
# train_seqs = tokenizer.texts_to_sequences(train_captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# Saving tokenizer
# tokenizer_json = tokenizer.to_json()
# with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
#     f.write(json.dumps(tokenizer_json, ensure_ascii=False))
# assert False

# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)

cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

MAX_LENGTH = max(len(t) for t in train_seqs)  # 49

print('Max sentence length :', MAX_LENGTH)

# Create training and validation sets
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector, cap_vector, test_size=0.1, random_state=0)

EPOCHS = 10
REPORT_PER_BATCH = 100
EPOCHS_TO_SAVE = 1
BATCH_SIZE = 80

BUFFER_SIZE = 20000
enc_layers = 6
dec_layers = 6
d_model = 512
dff = 2048
num_heads = 8
dropout_rate = 0.3
max_position_encodings = 256

vocab_size = num_words + 1
steps_per_epoch = len(img_name_train) // BATCH_SIZE


# Load the numpy files
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


# Train dataset
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
dataset.cache()

# Validation dataset
dataset_val = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
dataset_val = dataset_val.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset_val = dataset_val.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE)
dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Transformer model
transformer = Transformer(enc_layers, dec_layers, d_model, num_heads, dff,
                          vocab_size,
                          pe_input=max_position_encodings,
                          pe_target=max_position_encodings,
                          dropout_rate=dropout_rate)

learning_rate = CustomSchedule(d_model)
# learning_rate = lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     1e-4,
#     decay_steps=10000,
#     decay_rate=0.99)

# learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(1e-4, 10000, 0.3)
# learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, 10000, 0.5)

# plt.plot(learning_rate(tf.range(100000, dtype=tf.float32)))
# plt.ylabel("Learning Rate")
# plt.xlabel("Train Step")
# plt.show()
# assert False

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-9)
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1, reduction='none')

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


def loss_function(real, pred):
    # real.shape == (batch_size, seq_len)
    # pred.shape == (batch_size, seq_len, vocab_size)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    real = tf.one_hot(tf.cast(real, tf.int32), pred.shape[-1])
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2, output_type=tf.int32))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(tf.ones(shape=(tf.shape(inp)[0], tf.shape(inp)[1] * 2)))

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(tf.ones(shape=(tf.shape(inp)[0], tf.shape(inp)[1] * 2)))

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


train_step_signature = [
    tf.TensorSpec(shape=(None, 64, 2560), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32)
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    # inp.shape == (batch_size, 121, 2048)
    # tar.shape == (batch_size, seq_len)

    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))


# Checkpoints
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformoer=transformer,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=20)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1]) * EPOCHS_TO_SAVE
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)

print('Start Epoch = ', start_epoch)
print('Start training for {} epochs'.format(EPOCHS))
print('Batch Size = ', BATCH_SIZE)
print('Steps per epoch = ', steps_per_epoch)

loss_train_plot = []
loss_train_file = checkpoint_path + '/loss_train.npy'

if os.path.exists(loss_train_file):
    loss_train_plot = np.load(loss_train_file)

for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    current_epoch = start_epoch + epoch + 1

    for (batch, (img_tensor, target)) in enumerate(dataset):
        train_step(img_tensor, target)

        if batch % REPORT_PER_BATCH == 0:
            print('Epoch {} Batch {}/{} Loss {:.6f} Accuracy {:.6f}'.format(current_epoch, batch, steps_per_epoch, train_loss.result(), train_accuracy.result()))
            loss_train_plot = np.append(loss_train_plot, train_loss.result())

    if (epoch + 1) % EPOCHS_TO_SAVE == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(current_epoch, ckpt_save_path))

    print('Epoch {} Loss {:.6f} Accuracy {:.6f}'.format(current_epoch, train_loss.result(), train_accuracy.result()))
    loss_train_plot = np.append(loss_train_plot, train_loss.result())

    print('Time taken for {} epoch {} sec\n'.format(current_epoch, time.time() - start))

np.save(loss_train_file, loss_train_plot)

plt.plot(loss_train_plot, 'b', label='train loss')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()


def evaluate(image):
    img1, img2, _ = load_image(image)

    extracted_feature = extract_feature(img1[tf.newaxis, ...], img2[tf.newaxis, ...])

    # as the target is english, the first word to the transformer should be the english start token.
    decoder_input = [tokenizer.word_index['<start>']]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(extracted_feature, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions = transformer(extracted_feature,
                                  output,
                                  False,
                                  enc_padding_mask,
                                  combined_mask,
                                  dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token

        if predicted_id == tokenizer.word_index['<end>']:
            return tf.squeeze(output, axis=0)

        # concatenate the predicted_id to the output which is given to the decoder as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def decode(seq):
    predicted_caption = [tokenizer.index_word[i] for i in seq[1:] if i < vocab_size]

    return ' '.join(predicted_caption)


def decode_and_plot(image):
    result = evaluate(image)

    plt.imshow(np.array(Image.open(image)))
    plt.title(decode(result.numpy()))
    plt.show()

    print('Prediction Caption:', decode(result.numpy()))


# captions on the validation set
for it in range(10):
    rid = np.random.randint(0, len(img_name_val))
    image = img_name_val[rid]
    real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid][1:-1] if i not in [0]])

    print('Real Caption:', real_caption)
    decode_and_plot(image)

# assert False

image_url = 'https://tensorflow.org/images/surf.jpg'
# image_url = 'https://upload.wikimedia.org/wikipedia/commons/4/45/A_small_cup_of_coffee.JPG'
# image_url = 'https://post-phinf.pstatic.net/MjAxOTAyMTVfMjc2/MDAxNTUwMjA4NzE2MTIy.-Cae85qV570pF0FsWyoF2P4oEdooap7xS5vyfr3cGXUg.UaJFjECmhav26t5L985R9eg_cVS8zEDmyj_ihBrPR3wg.JPEG/3.jpg?type=w1200'
# image_url = 'https://raw.githubusercontent.com/yashk2810/Image-Captioning/master/images/frisbee.png'
# image_url = 'https://pds.joins.com/news/component/htmlphoto_mmdata/201910/30/f06c4fe8-dfa9-4ae4-af32-1aa0505d5cb1.jpg'
image_extension = image_url[-4:]
full_image_path = tf.keras.utils.get_file('image' + image_extension, origin=image_url)

# decode_and_plot(full_image_path)
decode_and_plot('C:/Users/Sopiro/Desktop/20200825/cut.png')
