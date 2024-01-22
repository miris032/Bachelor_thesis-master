import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def sin_generator(event_number):
    # 生成数据
    num_cycles = 4
    num_points_per_cycle = 50
    total_points = num_cycles * num_points_per_cycle
    x1 = np.linspace(100, 2 * np.pi * num_cycles, total_points)  # 生成0到6π的等间隔数据，以满足3个正弦周期
    x2 = np.linspace(50, 2 * np.pi * num_cycles, total_points)

    # 为两个正弦函数引入噪声
    noise1 = np.random.normal(0, 5, total_points)  # 第一个函数的噪声
    noise2 = np.random.normal(0, 5, total_points)  # 第二个函数的噪声

    # 正弦函数
    # acc
    # y1 = (np.sin(x1) + np.random.uniform(3.9, 5, total_points))
    '''y1 = (np.sin(x1)).astype(int) + np.random.uniform(15, 30, total_points).round(3)
    y2 = (np.sin(x1)).astype(int) + np.random.uniform(0.3, 7, total_points).round(3)
    y3 = (np.sin(x1)).astype(int) + np.random.uniform(0.5, 1, total_points).round(3)
    y4 = (np.sin(x1)).astype(int) + np.random.uniform(2, 5, total_points).round(3)
    y5 = (np.sin(x1)).astype(int) + np.random.uniform(1, 7, total_points).round(3)
    y6 = (np.sin(x1)).astype(int) + np.random.uniform(1, 5, total_points).round(3)
    y7 = (np.sin(x1)).astype(int) + np.random.uniform(0.5, 1.5, total_points).round(3)
    y8 = (np.sin(x1)).astype(int) + np.random.uniform(2, 5, total_points).round(3)
    y9 = (np.sin(x1)).astype(int) + np.random.uniform(0.3, 1.2, total_points).round(3)
    y10 = (np.sin(x1)).astype(int) + np.random.uniform(1, 5, total_points).round(3)'''

    # single stock price
    y1 = ((1.2 * np.sin(x1)).astype(int) + np.random.uniform(98.2, 99.2, total_points)).round(5)
    y2 = ((1.2 * np.sin(x1)).astype(int) + np.random.uniform(98.2, 99.2, total_points)).round(5)
    y3 = ((1.2 * np.sin(x1)).astype(int) + np.random.uniform(98.2, 99.2, total_points)).round(5)
    y4 = ((1.2 * np.sin(x1)).astype(int) + np.random.uniform(98.2, 99.2, total_points)).round(5)
    y5 = ((np.sin(x1)).astype(int) + np.random.uniform(93, 482, total_points)).astype(int)

    # useless
    y6 = (3 * np.cos(x1)).astype(int) + noise1 + 6.3
    y7 = (2 * np.cos(x1)).astype(int) + noise1 + 0.1
    y8 = (2 * np.cos(x1)).astype(int) + noise1 + 8.6
    y9 = (5 * np.cos(x1)).astype(int) + noise1 + 3
    y10 = (np.cos(x1)).astype(int) + noise1 + 63

    '''y1 = np.random.randint(50, 59, size=500)
    y2 = np.random.randint(265, 270, size=500)
    y3 = np.random.randint(110, 112, size=500)
    y4 = np.random.randint(199, 220, size=500)
    y5 = np.random.randint(63, 72, size=500)
    y6 = np.random.randint(84, 89, size=500)
    y7 = np.random.randint(6, 260, size=500)
    y8 = np.random.randint(16, 110, size=500)
    y9 = np.random.randint(10, 210, size=500)
    y10 = np.random.randint(63, 75, size=500)'''





    # 在两个正弦函数的第10到倒数第10个数据点之间引入n个同时刻发生的事件
    event_indices = np.random.randint(10, total_points - 10, size=event_number)
    print(event_indices)

    for event_index in event_indices:
        event_amplitude = np.random.uniform(1.5, 2.5)  # 随机事件增幅系数，保持在1.5到2.5之间
        y1[event_index:] *= event_amplitude  # 在事件发生后，第一个函数的振幅增加

    for event_index in event_indices:
        event_amplitude = np.random.uniform(3.5, 4)  # 随机事件增幅系数，保持在1.5到2.5之间
        y2[event_index:] *= event_amplitude  # 在事件发生后，第二个函数的振幅增加

    # 保存为CSV文件
    data = {'sin1': y1, 'sin2': y2, 'sin3': y3, 'sin4': y4, 'sin5': y5, 'sin6': y6, 'sin7': y7, 'sin8': y8, 'sin9': y9, 'sin10': y10}
    df = pd.DataFrame(data)
    df.to_csv('sin_with_noise_events.csv', index=False, header=False)
    print("saved as sin_with_noise_events.csv")

    label = np.zeros(total_points)
    label[event_indices] = 1
    marks = {'label': label}
    df2 = pd.DataFrame(marks)
    df2.to_csv('sin_with_noise_events_label.csv', index=False, header=False)
    print("saved as sin_with_noise_events_label.csv")


def read_and_plot_data(filename, label):
    # 读取CSV文件
    df = pd.read_csv(filename)
    # x = np.linspace(0, 2 * np.pi * 3, len(df))
    y1 = df.iloc[:, 0]
    y2 = df.iloc[:, 1]
    y3 = df.iloc[:, 2]
    y4 = df.iloc[:, 3]
    y5 = df.iloc[:, 4]
    y6 = df.iloc[:, 5]
    y7 = df.iloc[:, 6]
    y8 = df.iloc[:, 7]
    y9 = df.iloc[:, 8]
    y10 = df.iloc[:, 9]

    df0 = pd.read_csv(label)
    y0 = df0.iloc[:, 0]


    # 绘制图形
    plt.figure(figsize=(12, 5))
    plt.plot(y1, label='Sin Function 1', linewidth=0.3, alpha=0.5)
    plt.plot(y2, label='Sin Function 2', linewidth=0.3, alpha=0.5)
    plt.plot(y3, label='Sin Function 3', linewidth=0.3, alpha=0.5)
    plt.plot(y4, label='Sin Function 4', linewidth=0.3, alpha=0.5)
    plt.plot(y5, label='Sin Function 5', linewidth=0.3, alpha=0.5)
    plt.plot(y6, label='Sin Function 6', linewidth=0.3, alpha=0.5)
    plt.plot(y7, label='Sin Function 7', linewidth=0.3, alpha=0.5)
    plt.plot(y8, label='Sin Function 8', linewidth=0.3, alpha=0.5)
    plt.plot(y9, label='Sin Function 9', linewidth=0.3, alpha=0.5)
    plt.plot(y10, label='Sin Function 10', linewidth=0.3, alpha=0.5)

    for i, value in enumerate(y0, start=0):
        if value == 1:
            plt.axvline(x=i, color='gray', linestyle='--', linewidth=1, alpha=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Sine Functions with Noise and Multiple Events')
    plt.legend()
    #plt.grid(True)
    plt.show()


if __name__ == '__main__':
    sin_generator(0)
    # read_and_plot_data('sin_with_noise_events.csv', 'sin_with_noise_events_label.csv')
