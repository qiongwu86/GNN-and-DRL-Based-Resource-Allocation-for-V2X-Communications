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

dir = '../test/'
a = np.loadtxt(dir+'DDQN-GNN_rate.txt')
b = np.loadtxt(dir+'dqn_rate.txt')
c = np.loadtxt(dir+'Ashraf_rate.txt')
d = np.loadtxt(dir+'baseline_rate.txt')

small_five_size = 10.5
DDQN_GNN=a.tolist()
dqn=b.tolist()
Ashraf=c.tolist()
random = d.tolist()

x = [20, 40, 60, 80, 100]
#my_x_ticks = np.arange(4, 13, 2) # 原始数据有13个点，故此处为设置从0开始，间隔为1
fig=plt.figure()
#plt.tick_params(bottom=False)
#plt.xticks(my_x_ticks)
y_major_locator=MultipleLocator(25)
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
#plt.ylim(120,1500)
#box = plt.get_position()
#plt.set_position([box.x0, box.y0, box.width , box.height* 0.8])
plt.plot(x, DDQN_GNN, label='GNN-DDQN', marker='D',color='b')
plt.plot(x, dqn, label='DQN', marker='*', color='orange')
plt.plot(x, Ashraf, label='Method in Ashraf', marker='s',color='green')
plt.plot(x, random, label='Random Scheme', marker = 'o', color='r')
plt.ylabel('V2I Communication Rate', fontsize=small_five_size)
plt.xlabel('Number of Participating Vehicles', fontsize=small_five_size)
plt.legend(fontsize=10.5)
plt.grid(linestyle=':')
plt.xticks(fontsize=small_five_size)
plt.yticks(fontsize=small_five_size)
#plt.legend(loc='upper left', bbox_to_anchor=(0.99, 1.03),ncol=1)
#plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0)
#plt.subplots_adjust(bottom=0.15)
dir2=dir = '../结果图/'
plt.savefig(dir+'Fig_1_rate', dpi=600)
plt.show()