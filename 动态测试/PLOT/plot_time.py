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
c = np.loadtxt(dir+'num_veh.txt')

small_five_size = 10.5
num_veh=c.tolist()


x_data = np.linspace(0, 250, 5000)
#my_x_ticks = np.arange(4, 13, 2) # 原始数据有13个点，故此处为设置从0开始，间隔为1
fig=plt.figure()
#plt.tick_params(bottom=False)
#plt.xticks(my_x_ticks)
y_major_locator=MultipleLocator(20)
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(10, 150)
#box = plt.get_position()
#plt.set_position([box.x0, box.y0, box.width , box.height* 0.8])
plt.plot(x_data, num_veh, label='Number of Vehicles', color='green')
ax.set_xticks(np.arange(0, 251, 25))
plt.ylabel('Number of Vehicles in Environment', fontsize=small_five_size)
plt.xlabel('Time (s)', fontsize=small_five_size)
plt.legend(fontsize=10.5)
plt.grid(linestyle=':')
plt.xticks(fontsize=small_five_size)
plt.yticks(fontsize=small_five_size)
#plt.legend(loc='upper left', bbox_to_anchor=(0.99, 1.03),ncol=1)
#plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0)
#plt.subplots_adjust(bottom=0.15)
dir2=dir = '../test/'
plt.savefig(dir+'Fig_4_time2', dpi=600)
plt.show()