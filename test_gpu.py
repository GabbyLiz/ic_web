from tensorflow.python.client import device_lib
import tensorflow as tf

sess = tf.Session()

print(device_lib.list_local_devices())
