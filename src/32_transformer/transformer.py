import matplotlib.pyplot as plt
from layers import *

# pe = positional_encoding(50, 128)
#
# sample_encoder_layer = EncoderLayer(512, 8, 2048)
#
# sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)
#
# print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)

pos_encoding = positional_encoding(50, 1024)
print(pos_encoding.shape)

plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 1024))
plt.ylabel('Position')
plt.colorbar()
plt.show()
