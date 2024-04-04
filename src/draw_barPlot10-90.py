import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from matplotlib.lines import Line2D
import pandas as pd

root = Path(__file__).resolve().parent.parent


def get_drift_number_biobank(file, d_list):
    arr1 = []
    for i in range(1, 10):
        ol = int(d_list[0] * 0.1 * i)
        data = pd.read_pickle(
            f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_windowlength{d_list[0]}_overlap{ol}.pkl')
        content = list(data.values())[0]
        in_sum = np.sum(content)
        arr1.append(in_sum)
    print(f'd = {d_list[0]}: {arr1}')

    arr2 = []
    for i in range(1, 10):
        ol = int(d_list[1] * 0.1 * i)
        data = pd.read_pickle(
            f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_windowlength{d_list[1]}_overlap{ol}.pkl')
        content = list(data.values())[0]
        in_sum = np.sum(content)
        arr2.append(in_sum)
    print(f'd = {d_list[1]}: {arr2}')

    arr3 = []
    for i in range(1, 10):
        ol = int(d_list[2] * 0.1 * i)
        data = pd.read_pickle(
            f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_windowlength{d_list[2]}_overlap{ol}.pkl')
        content = list(data.values())[0]
        in_sum = np.sum(content)
        arr3.append(in_sum)
    print(f'd = {d_list[2]}: {arr3}')


def get_drift_number_stock(file):
    arr1 = []
    for i in range(1, 10):
        ol = int(30 * 0.1 * i)
        data = pd.read_pickle(
            f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_windowlength30_overlap{ol}.pkl')
        content = list(data.values())[0]
        in_sum = np.sum(content)
        arr1.append(in_sum)
    print(f'd = 30: {arr1}')

    arr2 = []
    for i in range(1, 10):
        ol = int(50 * 0.1 * i)
        data = pd.read_pickle(
            f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_windowlength50_overlap{ol}.pkl')
        content = list(data.values())[0]
        in_sum = np.sum(content)
        arr2.append(in_sum)
    print(f'd = 50: {arr2}')

    arr3 = []
    for i in range(1, 10):
        ol = int(100 * 0.1 * i)
        data = pd.read_pickle(
            f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_windowlength100_overlap{ol}.pkl')
        content = list(data.values())[0]
        in_sum = np.sum(content)
        arr3.append(in_sum)
    print(f'd = 100: {arr3}')


def draw_barPlot_biobank(data_array1, data_array2, data_array3):
    data_array1 = np.array(data_array1)
    data_array2 = np.array(data_array2)
    data_array3 = np.array(data_array3)

    # 生成 x 轴的刻度，从10%到90%
    x_ticks = np.arange(0.1, 1.0, 0.1)

    # 设置图形大小
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 计算相邻刻度之间的距离
    bar_width = x_ticks[1] - x_ticks[0]

    # 画第1个条形图
    axes[0].bar(x_ticks - bar_width / 2, data_array1, width=bar_width, color='lightblue', edgecolor='white')
    # axes[0].bar(x_ticks - bar_width / 2, data_array2, width=bar_width, color='pink', edgecolor='white', alpha=0.7)
    # axes[0].bar(x_ticks - bar_width / 2, data_array3, width=bar_width, color='lightgreen', edgecolor='white', alpha=0.7)
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
    axes[0].set_xlabel('overlap (a) in percent')
    axes[0].set_ylabel('detection number')
    axes[0].set_title('window length (d) = 50')

    # 画第2个条形图
    axes[1].bar(x_ticks - bar_width / 2, data_array2.flatten(), width=bar_width, color='lightblue', edgecolor='white')
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
    axes[1].set_xlabel('overlap (a) in percent')
    axes[1].set_ylabel('detection number')
    axes[1].set_title('window length (d) = 100')

    # 画第3个条形图
    axes[2].bar(x_ticks - bar_width / 2, data_array3, width=bar_width, color='lightblue', edgecolor='white')
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
    axes[2].set_xlabel('overlap (a) in percent')
    axes[2].set_ylabel('detection number')
    axes[2].set_title('window length (d) = 200')

    axes[2].legend()

    # 设置相同的y轴刻度范围
    min_y = min(data_array1.min(), data_array2.min(), data_array3.min())
    max_y = max(data_array1.max(), data_array2.max(), data_array3.max())
    for ax in axes:
        ax.set_ylim(min_y, max_y + 10)

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.4)

    # 显示图形
    plt.show()


def draw_barPlot_stock_BMW(data_array):
    data_array = np.array(data_array)

    # 生成 x 轴的刻度，从10%到90%
    x_ticks = np.arange(0.1, 1.0, 0.1)

    # 设置图形大小
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 计算相邻刻度之间的距离
    bar_width = (x_ticks[1] - x_ticks[0]) / 4  # 将条形图的宽度减小到原先的1/3

    # 画第1个条形图
    axes[0].bar(x_ticks - bar_width, data_array[0], width=bar_width, color='lightblue', edgecolor='white')
    axes[0].bar(x_ticks - bar_width * 2, data_array[1], width=bar_width, color='pink', edgecolor='white', alpha=0.75)
    axes[0].bar(x_ticks - bar_width * 3, data_array[2], width=bar_width, color='lightgreen', edgecolor='white', alpha=0.5)
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
    axes[0].set_xlabel('overlap (a) in percent', fontsize=16)
    axes[0].set_ylabel('detection number', fontsize=16)
    axes[0].set_title('window length (d) = 30', fontsize=20, pad=20)
    axes[0].tick_params(axis='both', which='major', labelsize=12)

    # 画第2个条形图
    axes[1].bar(x_ticks - bar_width, data_array[3], width=bar_width, color='lightblue', edgecolor='white')
    axes[1].bar(x_ticks - bar_width * 2, data_array[4], width=bar_width, color='pink', edgecolor='white', alpha=0.75)
    axes[1].bar(x_ticks - bar_width * 3, data_array[5], width=bar_width, color='lightgreen', edgecolor='white', alpha=0.5)
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
    axes[1].set_xlabel('overlap (a) in percent', fontsize=16)
    axes[1].set_ylabel('detection number', fontsize=16)
    axes[1].set_title('window length (d) = 50', fontsize=20, pad=20)
    axes[1].tick_params(axis='both', which='major', labelsize=12)

    # 画第3个条形图
    axes[2].bar(x_ticks - bar_width, data_array[6], width=bar_width, color='lightblue', edgecolor='white')
    axes[2].bar(x_ticks - bar_width * 2, data_array[7], width=bar_width, color='pink', edgecolor='white', alpha=0.75)
    axes[2].bar(x_ticks - bar_width * 3, data_array[8], width=bar_width, color='lightgreen', edgecolor='white', alpha=0.5)
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
    axes[2].set_xlabel('overlap (a) in percent', fontsize=16)
    axes[2].set_ylabel('detection number', fontsize=16)
    axes[2].set_title('window length (d) = 100', fontsize=20, pad=20)
    axes[2].tick_params(axis='both', which='major', labelsize=12)

    axes[2].legend([Line2D([0], [0], color='lightblue', lw=6),
                Line2D([0], [0], color='pink', lw=6, alpha=0.75),
                Line2D([0], [0], color='lightgreen', lw=6, alpha=0.5)],
               ['daily', 'hourly', 'minutely'], fontsize=16)

    # 设置相同的y轴刻度范围
    min_y = min(data_array.min(), data_array.min(), data_array.min())
    max_y = max(data_array.max(), data_array.max(), data_array.max())
    for ax in axes:
        ax.set_ylim(min_y, max_y + 10)

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.4)

    # 显示图形
    plt.show()



def draw_barPlot_stock(data_array1, data_array2, data_array3, d_list):

    # 生成 x 轴的刻度，从10%到90%
    x_ticks = np.arange(0.1, 1.0, 0.1)

    # 设置图形大小
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 计算相邻刻度之间的距离
    bar_width = x_ticks[1] - x_ticks[0]

    # 画第1个条形图
    axes[0].bar(x_ticks - bar_width / 2, data_array1, width=bar_width, color='lightblue', edgecolor='white')
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
    axes[0].set_xlabel('overlap (a) in percent', fontsize=16)
    axes[0].set_ylabel('detection number', fontsize=16)
    axes[0].set_title(f'window length (d) = {d_list[0]}', fontsize=20, pad=20)
    axes[0].tick_params(axis='both', which='major', labelsize=12)

    # 画第2个条形图
    axes[1].bar(x_ticks - bar_width / 2, data_array2, width=bar_width, color='lightblue', edgecolor='white')
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
    #axes[1].set_yticklabels(axes[1].get_yticks(), fontsize=12)
    axes[1].set_xlabel('overlap (a) in percent', fontsize=16)
    axes[1].set_ylabel('detection number', fontsize=16)
    axes[1].set_title(f'window length (d) = {d_list[1]}', fontsize=20, pad=20)
    axes[1].tick_params(axis='both', which='major', labelsize=12)

    # 画第3个条形图
    axes[2].bar(x_ticks - bar_width / 2, data_array3, width=bar_width, color='lightblue', edgecolor='white')
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'], fontsize=12)
    axes[2].set_xlabel('overlap (a) in percent', fontsize=16)
    axes[2].set_ylabel('detection number', fontsize=16)
    axes[2].set_title(f'window length (d) = {d_list[2]}', fontsize=20, pad=20)
    axes[2].tick_params(axis='both', which='major', labelsize=12)

    # 设置相同的y轴刻度范围
    data_array1 = np.array(data_array1)
    data_array2 = np.array(data_array2)
    data_array3 = np.array(data_array3)
    min_y = min(data_array1.min(), data_array2.min(), data_array3.min())
    max_y = max(data_array1.max(), data_array2.max(), data_array3.max())
    for ax in axes:
        ax.set_ylim(min_y, max_y + 10)

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.4)

    # 显示图形
    plt.show()


if __name__ == '__main__':

    # dataset1-timeSeries_filled_binned
    '''get_drift_number_biobank('dataset1-timeSeries_filled_binned', [50, 100, 200])
    draw_barPlot_stock([0, 0, 0, 0, 0, 0, 5, 12, 76],
                       [0, 0, 0, 1, 1, 2, 5, 12, 44],
                       [0, 0, 2, 2, 3, 5, 6, 12, 31],
                       [50, 100, 200])'''

    # accs
    '''get_drift_number_biobank('accs_filled_binned', [50, 100, 200])
    draw_barPlot_stock([16, 16, 17, 23, 19, 31, 46, 98, 207],
                       [7, 9, 12, 16, 12, 20, 26, 47, 116],
                       [0, 0, 0, 2, 7, 11, 14, 22, 43],
                       [50, 100, 200])'''






    '''print('daily: ')
    get_drift_number_stock('BMW.DE_daily_binned50')
    print()
    print('hourly: ')
    get_drift_number_stock('BMW.DE_hourly_binned50')
    print()
    print('minutely: ')
    get_drift_number_stock('BMW.DE_minutely_binned50')'''

    # daily
    '''draw_barPlot_stock([0, 1, 1, 0, 1, 1, 2, 6, 51],
                 [0, 0, 0, 2, 3, 4, 6, 6, 27],
                 [1, 0, 0, 0, 1, 1, 2, 4, 18])'''

    # hourly
    '''draw_barPlot_stock([0, 0, 0, 1, 0, 1, 3, 8, 48],
                         [0, 0, 0, 0, 0, 0, 3, 3, 25],
                         [0, 0, 0, 0, 0, 0, 0, 3, 8])'''

    # minutely
    '''draw_barPlot_stock([0, 0, 0, 0, 0, 0, 0, 0, 11],
                             [0, 0, 0, 0, 0, 0, 0, 0, 11],
                             [0, 0, 0, 0, 0, 0, 0, 1, 4])'''

    draw_barPlot_stock_BMW([[0, 1, 1, 0, 1, 1, 2, 6, 51], [0, 0, 0, 2, 3, 4, 6, 6, 27], [1, 0, 0, 0, 1, 1, 2, 4, 18],

                        [0, 0, 0, 1, 0, 1, 3, 8, 48], [0, 0, 0, 0, 0, 0, 3, 3, 25], [0, 0, 0, 0, 0, 0, 0, 3, 8],

                        [0, 0, 0, 0, 0, 0, 0, 0, 11], [0, 0, 0, 0, 0, 0, 0, 0, 11], [0, 0, 0, 0, 0, 0, 0, 1, 4]])

    # Open_hourly
    '''get_drift_number_biobank('Open_hourly_binned', [30, 50, 100])
    draw_barPlot_stock([0, 0, 0, 0, 1, 3, 4, 17, 120],
                       [0, 0, 0, 0, 2, 2, 2, 11, 60],
                       [0, 0, 0, 0, 0, 0, 1, 5, 29],
                       [30, 50, 100])'''
