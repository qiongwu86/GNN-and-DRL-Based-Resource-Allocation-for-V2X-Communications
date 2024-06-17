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
a = 1-np.loadtxt(dir+'p_epoch.txt')
b = np.loadtxt(dir+'rate_epoch.txt')


small_five_size = 10.5
p =a.tolist()
rate =b.tolist()


x = [0, 2000, 4000, 6000, 8000, 10000]
#my_x_ticks = np.arange(4, 13, 2) # 原始数据有13个点，故此处为设置从0开始，间隔为1
fig, ax1 = plt.subplots()
#plt.tick_params(bottom=False)
#plt.xticks(my_x_ticks)
y_major_locator = MultipleLocator(0.01)
ax1.yaxis.set_major_locator(y_major_locator)
#plt.ylim(120,1500)
#box = plt.get_position()
#plt.set_position([box.x0, box.y0, box.width , box.height* 0.8])
ax1.plot(x, p, label='V2V Success Rate', marker='D', color='b')
ax1.set_ylabel('V2V Communication Success Rate', fontsize=10.5)
ax1.set_xlabel('Number of Epochs', fontsize=10.5)

ax2 = ax1.twinx()
ax2.yaxis.set_major_locator(MultipleLocator(10))
ax2.plot(x, rate, label='V2I Rate', marker='*', color='orange')
ax2.set_ylabel('V2I Communication Rate', fontsize=10.5)


# 设置刻度字体大小
ax1.tick_params(axis='both', which='major', labelsize=10.5)
ax2.tick_params(axis='y', which='major', labelsize=10.5)
# plt.ylabel('epoch数目', fontsize=small_five_size)
# plt.xlabel('参与车辆数', fontsize=small_five_size)
ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10.5)
ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.90), fontsize=10.5)

plt.grid(linestyle=':')
plt.xticks(fontsize=small_five_size)
plt.yticks(fontsize=small_five_size)
#plt.legend(loc='upper left', bbox_to_anchor=(0.99, 1.03),ncol=1)
#plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0)
#plt.subplots_adjust(bottom=0.15)
dir2=dir = '../结果图/'
plt.savefig(dir+'Fig_3_epoch2', dpi=600)
plt.show()