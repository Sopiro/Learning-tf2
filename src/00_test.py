import tensorflow as tf

print(tf.test.is_gpu_available())
print(tf.test.is_built_with_cuda())
print(tf.test.gpu_device_name())
