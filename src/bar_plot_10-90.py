import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 生成两个示例的数据数组
    data_array1 = np.array([0, 0, 0, 1, 0,  0, 5, 11, 73])
    data_array2 = np.array([0, 0, 1, 2, 3,  5, 6, 12, 32])

    # 生成 x 轴的刻度，从10%到90%
    x_ticks = np.arange(0.1, 1.0, 0.1)

    # 设置图形大小
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 计算相邻刻度之间的距离
    bar_width = x_ticks[1] - x_ticks[0]

    # 画第一个条形图
    axes[0].bar(x_ticks - bar_width / 2, data_array1, width=bar_width, color='lightblue', edgecolor='white')
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
    axes[0].set_xlabel('overlap (in percent)')
    axes[0].set_ylabel('number of detected drift points')
    axes[0].set_title('d = 50')

    # 画第二个条形图
    axes[1].bar(x_ticks - bar_width / 2, data_array2, width=bar_width, color='lightblue', edgecolor='white')
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
    axes[1].set_xlabel('overlap (in percent)')
    axes[1].set_ylabel('number of detected drift points')
    axes[1].set_title('d = 200')

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.25)

    # 显示图形
    plt.show()
