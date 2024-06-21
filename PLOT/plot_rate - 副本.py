import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

plt.rcParams['font.sans-serif']=['Arial'] #显示中文
plt.rcParams['axes.unicode_minus'] = False
plt.rc("font",family="Arial",size="10.5")

def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# 读取数据
dir = '../test/'
a = np.loadtxt(dir + 'DDQN-GNN_rate.txt')
b = np.loadtxt(dir + 'dqn_rate.txt')
c = np.loadtxt(dir + 'Ashraf_rate.txt')
d = np.loadtxt(dir + 'baseline_rate.txt')

small_five_size = 10.5
DDQN_GNN = a.tolist()
dqn = b.tolist()
Ashraf = c.tolist()
random = d.tolist()

x = np.array([20, 40, 60, 80, 100])
width = 3.0  # 每个柱子的宽度
group_gap = width * 0.1  # 每组柱子之间的间隙比例

fig, ax = plt.subplots()

# 设置柱状图
bar1 = ax.bar(x - 1.5 * (width + group_gap / 3), DDQN_GNN, width, label='GNN-DDQN', color='#1f77b4')
bar2 = ax.bar(x - 0.5 * (width + group_gap / 3), dqn, width, label='DQN', color='#ff7f0e')
bar3 = ax.bar(x + 0.5 * (width + group_gap / 3), Ashraf, width, label='Method in Ashraf', color='#2ca02c')
bar4 = ax.bar(x + 1.5 * (width + group_gap / 3), random, width, label='Random Scheme', color='#d62728')

# 设置刻度和标签
y_major_locator = MultipleLocator(25)
ax.yaxis.set_major_locator(y_major_locator)
ax.set_ylim(0, ax.get_ylim()[1])  # y轴从20开始
ax.set_xlabel('Number of Participating Vehicles', fontsize=small_five_size)
ax.set_ylabel('V2I Communication Rate', fontsize=small_five_size)
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in x])
ax.legend(fontsize=10.5)
ax.grid(linestyle=':')

# 设置刻度字体大小
plt.xticks(fontsize=small_five_size)
plt.yticks(fontsize=small_five_size)

# 保存和显示图形
dir2 = '../结果图/'
plt.savefig(dir2 + 'Fig_1_rate1', dpi=600)
plt.show()