import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 生成两个示例的数据数组
    data_array1 = np.array([0, 0, 0, 1, 0, 0, 5, 11, 73])
    data_array2 = np.array([[0, 0, 1, 1, 0, 1, 6, 12, 39]])
    data_array3 = np.array([0, 0, 1, 2, 3, 5, 6, 12, 32])

    # 生成 x 轴的刻度，从10%到90%
    x_ticks = np.arange(0.1, 1.0, 0.1)

    # 设置图形大小
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 计算相邻刻度之间的距离
    bar_width = x_ticks[1] - x_ticks[0]

    # 画第1个条形图
    axes[0].bar(x_ticks - bar_width / 2, data_array1, width=bar_width, color='lightblue', edgecolor='white')
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
    axes[0].set_xlabel('a (overlap) in percent')
    axes[0].set_ylabel('number of detected drift points')
    axes[0].set_title('d (window length) = 50')
    axes[0].set_ylim(0.25, max(data_array1.max(), data_array2.max(), data_array3.max()))

    # 画第2个条形图
    axes[1].bar(x_ticks - bar_width / 2, data_array2.flatten(), width=bar_width, color='lightblue', edgecolor='white')
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
    axes[1].set_xlabel('a (overlap) in percent')
    axes[1].set_ylabel('number of detected drift points')
    axes[1].set_title('d (window length) = 100')
    axes[1].set_ylim(0.25, max(data_array1.max(), data_array2.max(), data_array3.max()))

    # 画第3个条形图
    axes[2].bar(x_ticks - bar_width / 2, data_array3, width=bar_width, color='lightblue', edgecolor='white')
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
    axes[2].set_xlabel('a (overlap) in percent')
    axes[2].set_ylabel('number of detected drift points')
    axes[2].set_title('d (window length) = 200')
    axes[2].set_ylim(0.25, max(data_array1.max(), data_array2.max(), data_array3.max()))

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.4)

    # 显示图形
    plt.show()
