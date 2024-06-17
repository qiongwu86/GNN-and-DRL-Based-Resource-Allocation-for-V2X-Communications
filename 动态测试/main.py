from __future__ import division, print_function
import random
import tensorflow as tf
from agent import Agent
from Environment import *
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 这会告诉TensorFlow忽略GPU

# 创建命令行解析器
parser = argparse.ArgumentParser()

# Model
parser.add_argument('--model', type=str, default='m1', help='Type of model')
parser.add_argument('--dueling', action='store_true', help='Whether to use dueling deep q-network')
parser.add_argument('--double_q', action='store_true', help='Whether to use double q-learning')

# Environment
parser.add_argument('--env_name', type=str, default='Breakout-v0', help='The name of gym environment to use')
parser.add_argument('--action_repeat', type=int, default=4, help='The number of action to be repeated')

# Etc
parser.add_argument('--use_gpu', action='store_true', help='Whether to use gpu or not')
parser.add_argument('--gpu_fraction', type=str, default='1/1', help='idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
parser.add_argument('--display', action='store_true', help='Whether to display the game screen or not')
parser.add_argument('--is_train', action='store_true', help='Whether to do training or testing')
parser.add_argument('--random_seed', type=int, default=123, help='Value of random seed')

FLAGS = parser.parse_args()

# Set random seed
tf.random.set_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")


def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)
  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction


# GPU配置
if FLAGS.use_gpu:
  gpu_fraction = calc_gpu_fraction(FLAGS.gpu_fraction)
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      # 设置GPU显存按需分配
      tf.config.experimental.set_memory_growth(gpus[0], True)
      if gpu_fraction < 1:
        # 分配一部分显存给进程使用
        memory_limit = gpu_fraction * tf.config.experimental.get_memory_info(gpus[0])['total']
        tf.config.experimental.set_virtual_device_configuration(
          gpus[0],
          [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
        )
    except RuntimeError as e:
      print(e)


def main():
  # ... 保持原有逻辑不变 ...
  up_lanes = [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]
  down_lanes = [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
                750 - 3.5 / 2]
  left_lanes = [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]
  right_lanes = [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,
                 1299 - 3.5 / 2]
  width = 750
  height = 1299
  Env = Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height)
  # Env.new_random_game()
  Env.create_dynamic_env()
  agent = Agent([], Env)  # 注意，在 TensorFlow 2.x 中，不再需要传递会话(sess)
  # with tf.device('/cpu:0'):
  #agent.train()
  agent.play_for_dynamic_environment()
  # agent.play()


if __name__ == '__main__':
  main()


