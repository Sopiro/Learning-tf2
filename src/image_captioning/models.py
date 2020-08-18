import tensorflow as tf


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.W3 = tf.keras.layers.Dense(units)
        self.W4 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size) == (batch_size, units)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis1 = tf.expand_dims(hidden[0], 1)
        hidden_with_time_axis2 = tf.expand_dims(hidden[1], 1)
        hidden_with_time_axis3 = tf.expand_dims(hidden[2], 1)

        # Bahdanau Score function = tanh( W1 * key + W2 * query )
        # score shape == (batch_size, 64, hidden_size), Matrix broadcasting works in here
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis1) + self.W3(hidden_with_time_axis2) + self.W4(hidden_with_time_axis3))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, embedding_dim)
        context_vector = attention_weights * features  # Matrix broadcasting works in here
        # context_vector = tf.reduce_sum(context_vector, axis=1)  # todo: Don't collapse weighted features into one vector
        context_vector = tf.reshape(context_vector, (context_vector.shape[0], -1))

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru1 = tf.keras.layers.GRU(self.units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.gru2 = tf.keras.layers.GRU(self.units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.gru3 = tf.keras.layers.GRU(self.units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + embedding_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        # output shape = (batch_size, 1, hidden_size)
        # state shape = (batch_size, hidden_size)
        x, state1 = self.gru1(x)
        x, state2 = self.gru2(x)
        x, state3 = self.gru3(x)

        # shape == (batch_size, 1, hidden_size)
        x = self.fc1(x)

        # x shape == (batch_size, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # shape == (batch_size, vocab)
        x = self.fc2(x)

        return x, [state1, state2, state3], attention_weights

    def reset_state(self, batch_size):
        return [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))]
