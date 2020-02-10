import tensorflow as tf
import numpy as np
import re
import matplotlib.pyplot as plt

path_to_train_file = tf.keras.utils.get_file('train.txt',
                                             'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt')
path_to_test_file = tf.keras.utils.get_file('train.txt',
                                            'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt')

train_text = open(path_to_train_file, 'rb').read().decode(encoding='utf-8')
test_text = open(path_to_test_file, 'rb').read().decode(encoding='utf-8')

# print(train_text[:300])

train_Y = np.array([[int(row.split('\t')[2])] for row in train_text.split('\n')[1:] if row.count('\t') > 0])
test_Y = np.array([[int(row.split('\t')[2])] for row in test_text.split('\n')[1:] if row.count('\t') > 0])


# From https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
def clean_str(string):
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\{2,}", "\'", string)
    string = re.sub(r"\'", "", string)

    return string.lower()


# Split line by line, clean the sentences and split sentence by whitespace
train_text_X = [row.split('\t')[1] for row in train_text.split('\n')[1:] if row.count('\t') > 0]
train_text_X = [clean_str(sentence) for sentence in train_text_X]

sentences = [sentence.split(' ') for sentence in train_text_X]

# for i in range(5):
#     print(sentences[i])
# assert False

# Cut words to 5 length
sentences_new = []
for sentence in sentences:
    sentences_new.append([word[:5] for word in sentence][:25])

sentences = sentences_new

# for i in range(5):
#     print(sentences[i])
# assert False

#  num_words:
#   the maximum number of words to keep, based
#   on word frequency. Only the most common num_words-1 words will be kept
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(sentences)
train_X = tokenizer.texts_to_sequences(sentences)
train_X = tf.keras.preprocessing.sequence.pad_sequences(train_X, padding='post')

# print(train_X[:5])
# for i in range(1, 10):
#     print(tokenizer.index_word[i])
# assert False

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=20000, output_dim=300, input_length=25),
    tf.keras.layers.LSTM(units=50),
    tf.keras.layers.Dense(units=2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_X, train_Y, epochs=5, batch_size=128, validation_split=0.2)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim(0.7)
plt.show()

test_text_X = [row.split('\t')[1] for row in test_text.split('\n')[1:] if row.count('\t') > 0]
test_text_X = [clean_str(sentence) for sentence in test_text_X]

sentences = [sentence.split(' ') for sentence in test_text_X]

sentences_new = []
for sentence in sentences:
    sentences_new.append([word[:5] for word in sentence][:25])

sentences = sentences_new

test_X = tokenizer.texts_to_sequences(sentences)
test_X = tf.keras.preprocessing.sequence.pad_sequences(test_X, padding='post')

model.evaluate(test_X, test_Y)

test_sentence = '내가 만들어도 이것보단 재밌겠다'
test_sentence = test_sentence.split(' ')
test_sentences = []
now_sentence = []

for word in test_sentence:
    now_sentence.append(word)
    test_sentences.append(now_sentence[:])

test_X_1 = tokenizer.texts_to_sequences(test_sentences)
test_X_1 = tf.keras.preprocessing.sequence.pad_sequences(test_X_1, padding='post', maxlen=25)

prediction = model.predict(test_X_1)

# [accuracy, loss]
for idx, sentence in enumerate(test_sentences):
    print(prediction[idx], sentence)
