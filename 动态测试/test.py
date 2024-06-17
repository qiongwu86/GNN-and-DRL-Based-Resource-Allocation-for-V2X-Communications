import tensorflow as tf

print(tf.__version__)  # 查看tensorflow版本
print(tf.__path__)  # 查看tensorflow安装路径

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)

