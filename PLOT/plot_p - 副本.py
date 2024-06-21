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
a = 1 - np.loadtxt(dir + 'DDQN-GNN_p.txt')
b = 1 - np.loadtxt(dir + 'dqn_p.txt')
c = 1 - np.loadtxt(dir + 'Ashraf_p.txt')
d = 1 - np.loadtxt(dir + 'baseline_p.txt')

small_five_size = 10.5
DDQN_GNN = a.tolist()
dqn = b.tolist()
Ashraf = c.tolist()
random = d.tolist()

x = np.array([20, 40, 60, 80, 100])
width = 3.0  # 每个柱子的宽度
group_gap = width * 0.1  # 每组柱子之间的间隙比例
# 调整图形的宽高比
fig, ax = plt.subplots()
# 设置柱状图，使用好看的颜色搭配
bar1 = ax.bar(x - 1.5 * (width + group_gap / 3), DDQN_GNN, width, label='GNN-DDQN', color='#1f77b4')  # 蓝色
bar2 = ax.bar(x - 0.5 * (width + group_gap / 3), dqn, width, label='DQN', color='#ff7f0e')  # 橙色
bar3 = ax.bar(x + 0.5 * (width + group_gap / 3), Ashraf, width, label='Method in Ashraf', color='#2ca02c')  # 绿色
bar4 = ax.bar(x + 1.5 * (width + group_gap / 3), random, width, label='Random Scheme', color='#d62728')  # 红色


# 设置刻度和标签
y_major_locator = MultipleLocator(0.02)
ax.yaxis.set_major_locator(y_major_locator)
ax.set_ylim(0.8, 1)  # 设置 y 轴的范围
ax.set_xlabel('Number of Participating Vehicles', fontsize=small_five_size)
ax.set_ylabel('V2V Communication Success Rate', fontsize=small_five_size)
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in x])
ax.legend(fontsize=10.5)
ax.grid(linestyle=':')

# 设置图例位置
ax.legend(fontsize='small', loc='upper right', bbox_to_anchor=(1, 1), handlelength=2, borderpad=0.5)

# 设置刻度字体大小
plt.xticks(fontsize=small_five_size)
plt.yticks(fontsize=small_five_size)

# 保存和显示图形
dir2 = '../结果图/'
plt.savefig(dir2 + 'Fig_1_p1', dpi=600)
plt.show()