from ch30_transformer_captioning.layers import *


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, dropout_rate=0.1):
        """
        Transformer encoder

        :param num_layers: Number of encoder layers
        :param d_model: Model's dimension, Word's embedding dimension
        :param num_heads: Number of heads for Multi-head attention
        :param dff: Feed forward network units
        :param maximum_position_encoding: This can be the maximum length of sequence or sentence
        :param dropout_rate: Dropout rate
        """
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dense1 = tf.keras.layers.Dense(d_model)
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.relu = tf.keras.layers.ReLU()
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        # Stacking encoder layers
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        # x.shape == (batch_size, image_feature_len:64, feature_dim:2048+512=2560)
        seq_len = tf.shape(x)[1]

        xi = self.dense1(x[:, :, :2048]) + self.pos_encoding[:, :seq_len, :]  # shape == (batch_size, image_feature_len, d_model)
        xf = self.dense2(x[:, :, 2048:])  # shape == (batch_size, image_feature_len, d_model)

        x = tf.concat([xi, xf], axis=1)  # shape == (batch_size, image_feature_len*2:128, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # Scaling for normalization

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len*2, d_model)


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
        # x.shape == (batch_size, seq_len)
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
    def __init__(self, enc_layers, dec_layers, d_model, num_heads, dff, target_vocab_size, pe_input, pe_target, dropout_rate=0.1):
        """
        Transformer network

        :param enc_layers: Number of encoder, decoder stacks
        :param dec_layers: Number of decoder, decoder stacks
        :param d_model: Model's dimension, Word's embedding dimension
        :param num_heads: Number of heads for Multi-head attention
        :param dff: Point-wise feed forward network units
        :param target_vocab_size: Target space vocabulary size
        :param pe_input: Input's maximum positional encoding dimension
        :param pe_target: Target's maximum positional encoding dimension
        :param dropout_rate: Dropout rate
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(enc_layers, d_model, num_heads, dff, pe_input, dropout_rate)
        self.decoder = Decoder(dec_layers, d_model, num_heads, dff, target_vocab_size, pe_target, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, input_data, target_data, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # input_data.shape == (batch_size, feature_len:64, feature_dim:2048+512)
        # target_data.shape == (batch_size, seq_len)
        # enc_output.shape == (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(input_data, training, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder(target_data, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=8000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * 0.6 * tf.math.minimum(arg1 + 1e-9, arg2 + 1e-8)
