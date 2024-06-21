import os

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
a = 1 - np.loadtxt(dir + 'Fail_percent.txt')
b = np.loadtxt(dir+'V2I_Rate.txt')

small_five_size = 10.5
num_veh=c.tolist()



# 示例数据
v2v_success_rate = a
v2i_rate = b
vehicle_count = c

# 将数据分成五份，每份1000个数据点
data_chunks_v2v = [v2v_success_rate[i:i + 1000] for i in range(0, 5000, 1000)]
data_chunks_v2i = [v2i_rate[i:i + 1000] for i in range(0, 5000, 1000)]
data_chunks_vehicle = [vehicle_count[i:i + 1000] for i in range(0, 5000, 1000)]

# 创建一个图形并设置大小
fig, ax1 = plt.subplots()

# 创建第二个Y轴
ax2 = ax1.twinx()

# 为每组数据设置不同的颜色
colors = ['lightblue', 'lightgreen', 'lightslategray']

# 绘制每组数据的箱形图
positions = np.arange(1, 6)  # X轴上的位置
width = 0.2  # 每个箱形图的宽度

# 在每个位置绘制三个箱形图（分别对应V2V成功率、V2I速率和车辆数目）
for i in range(5):
    boxprops_v2v = dict(facecolor=colors[0])
    boxprops_v2i = dict(facecolor=colors[1])
    boxprops_vehicle = dict(facecolor=colors[2])

    ax1.boxplot(data_chunks_v2i[i], positions=[positions[i] - width], widths=width, patch_artist=True,
                boxprops=boxprops_v2i)
    ax1.boxplot(data_chunks_vehicle[i], positions=[positions[i]], widths=width, patch_artist=True,
                boxprops=boxprops_vehicle)
    ax2.boxplot(data_chunks_v2v[i], positions=[positions[i] + width], widths=width, patch_artist=True,
                boxprops=boxprops_v2v)

# 设置X轴刻度
ax1.set_xticks(positions)
ax1.set_xticklabels([f'Chunk {i + 1}' for i in range(5)])

# 设置标题和轴标签
ax1.set_title('Boxplots of V2V Success Rate, V2I Rate, and Vehicle Count')
ax1.set_xlabel('Data Chunk')
ax1.set_ylabel('V2I Rate and Vehicle Count')
ax2.set_ylabel('V2V Success Rate')

# 设置次Y轴的刻度间隔
ax2.set_yticks(np.arange(0.8, 1.02, 0.02))

# 添加图例
handles = [
    plt.Line2D([0], [0], color='lightblue', lw=4, label='V2V Success Rate'),
    plt.Line2D([0], [0], color='lightgreen', lw=4, label='V2I Rate'),
    plt.Line2D([0], [0], color='lightslategray', lw=4, label='Vehicle Count'),
]
ax1.legend(handles=handles, loc='upper right')  # 将图例放置在右上角

# 显示图形
ax1.grid(True)
# 保存图形
plt.savefig(dir + 'Boxplots of V2V Success Rate, V2I Rate, and Vehicle Count.png', dpi=600)
plt.show()