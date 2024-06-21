import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 假设数据已经加载好
dir = '../test/'
a = 1 - np.loadtxt(dir + 'Fail_percent.txt')
b = np.loadtxt(dir + 'V2I_Rate.txt')
num_veh = np.loadtxt(dir + 'num_veh.txt')

# 转换为列表
p = a.tolist()
rate = b.tolist()
num_veh_list = num_veh.tolist()

# 时间数据
x_data = np.linspace(0, 250, 5000)

# 设置字体大小
small_five_size = 10  # 调整内容字体大小
y_axis_font_size = 8  # 调整Y轴刻度字体大小

# 创建图形
fig, ax1 = plt.subplots(figsize=(10, 6))

# 设置第一个y轴刻度间隔
y_major_locator = MultipleLocator(0.01)
ax1.yaxis.set_major_locator(y_major_locator)

# 绘制V2V成功率
ax1.plot(x_data, p, label='Instantaneous V2V Success Rate', color='blue', linewidth=2)
ax1.set_xticks(np.arange(0, 251, 25))
ax1.set_ylabel('V2V Communication Success Rate', fontsize=small_five_size, color='blue')
ax1.set_xlabel('Time (s)', fontsize=small_five_size)
ax1.tick_params(axis='y', which='major', labelcolor='blue', labelsize=y_axis_font_size)

# 设置第二个y轴
ax2 = ax1.twinx()
ax2.yaxis.set_major_locator(MultipleLocator(10))

# 绘制V2I通信速率
ax2.plot(x_data, rate, label='Instantaneous V2I Rate', color='orange', linewidth=2)
ax2.set_ylabel('V2I Communication Rate', fontsize=small_five_size, color='orange')
ax2.tick_params(axis='y', which='major', labelcolor='orange', labelsize=y_axis_font_size)

# 设置第三个y轴
ax3 = ax1.twinx()
# 将第三个Y轴靠近第二个Y轴
ax3.spines["right"].set_position(("outward", 40))  # 调整间距
ax3.yaxis.set_major_locator(MultipleLocator(10))

# 绘制车辆数目
ax3.plot(x_data, num_veh_list, label='Number of Vehicles', color='green', linewidth=2)
ax3.set_ylabel('Number of Vehicles', fontsize=small_five_size, color='green')
ax3.tick_params(axis='y', which='major', labelcolor='green', labelsize=y_axis_font_size)

# 设置图例位置和字体大小
ax1.legend(loc='upper right', bbox_to_anchor=(0.98, 0.65), fontsize='small')
ax2.legend(loc='upper right', bbox_to_anchor=(0.98, 0.57), fontsize='small')
ax3.legend(loc='upper right', bbox_to_anchor=(0.98, 0.49), fontsize='small')

# 添加网格
plt.grid(linestyle=':')
fig.tight_layout(pad=5.0)
# 保存图形
plt.savefig(dir + 'Fig_dynamic_result2_3y_improved.png', dpi=600)

# 显示图形
plt.show()
