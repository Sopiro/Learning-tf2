import tensorflow as tf


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size) == (batch_size, units)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # Bahdanau Score function = tanh( W1 * key + W2 * query )
        # score shape == (batch_size, 64, hidden_size), Matrix broadcasting works in here
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, embedding_dim)
        context_vector = attention_weights * features  # Matrix broadcasting works in here
        context_vector = tf.reduce_sum(context_vector, axis=1)  # Try no collapsing weighted features into one vector

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, feature_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, feature_dim)
        self.fc = tf.keras.layers.Dense(feature_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, rnn_units, fc_units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.rnn_units = rnn_units
        self.fc_units = fc_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru1 = tf.keras.layers.GRU(self.rnn_units,
                                        return_sequences=True,
                                        return_state=False,
                                        recurrent_initializer='glorot_uniform')
        self.gru2 = tf.keras.layers.GRU(self.rnn_units,
                                        return_sequences=True,
                                        return_state=False,
                                        recurrent_initializer='glorot_uniform')
        self.gru3 = tf.keras.layers.GRU(self.rnn_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.fc_units, kernel_regularizer='l2')
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.fc2 = tf.keras.layers.Dense(vocab_size, kernel_regularizer='l2')

        self.attention = BahdanauAttention(self.rnn_units)

    def call(self, x, features, hidden, training=False):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + embedding_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        # output shape = (batch_size, 1, hidden_size)
        # state shape = (batch_size, hidden_size)
        x = self.gru1(x)
        x = self.gru2(x)
        x, state = self.gru3(x)

        # shape == (batch_size, 1, hidden_size)
        x = self.fc1(x)

        # x shape == (batch_size, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # Using drop out
        x = self.dropout(x, training=training)

        # shape == (batch_size, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.rnn_units))
