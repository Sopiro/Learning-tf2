from layers import *


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        """
        Transformer encoder

        :param num_layers: Number of encoder layers
        :param d_model: Model's dimension, Word's embedding dimension
        :param num_heads: Number of heads for Multi-head attention
        :param dff: Feed forward network units
        :param input_vocab_size: Input space vocabulary size
        :param maximum_position_encoding: This can be the maximum length of sequence or sentence
        :param dropout_rate: Dropout rate
        """
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        # Stacking encoder layers
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # Adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # Scaling for normalization
        x += self.pos_encoding[:, :seq_len, :]  # Broadcasting works here

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        """
        Transformer decoder

        :param num_layers: Number of decoder layers
        :param d_model: Model's dimension, Word's embedding dimension
        :param num_heads: Number of heads for Multi-head attention
        :param dff: Point-wise feed forward network units
        :param target_vocab_size: Target space vocabulary size
        :param maximum_position_encoding: This can be the maximum length of sequence or sentence
        :param dropout_rate: Dropout rate
        """
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # Scaling for normalization
        x += self.pos_encoding[:, :seq_len, :]  # Broadcasting works here

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, dropout_rate=0.1):
        """
        Transformer network

        :param num_layers: Number of encoder, decoder stacks
        :param d_model: Model's dimension, Word's embedding dimension
        :param num_heads: Number of heads for Multi-head attention
        :param dff: Point-wise feed forward network units
        :param input_vocab_size: Input space vocabulary size
        :param target_vocab_size: Target space vocabulary size
        :param pe_input: Input's maximum positional encoding dimension
        :param pe_target: Target's maximum positional encoding dimension
        :param dropout_rate: Dropout rate
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, input_data, target_data, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # enc_output.shape == (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(input_data, training, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder(target_data, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output
