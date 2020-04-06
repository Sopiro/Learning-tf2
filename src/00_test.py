import tensorflow as tf
import time

print(tf.__version__)
print(tf.test.is_gpu_available())
print(tf.test.is_built_with_cuda())
print(tf.test.gpu_device_name())


# CPU vs. GPU from tensorflow official tutorial page.

def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)

    result = time.time() - start

    print("10 loops: {:0.2f}ms".format(1000 * result))


# CPU에서 강제 실행합니다.
print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

# GPU #0가 이용가능시 GPU #0에서 강제 실행합니다.
if tf.test.is_gpu_available():
    print("On GPU:")
    with tf.device("GPU:0"):  # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
        x = tf.random.uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)
