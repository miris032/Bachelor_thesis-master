import csv
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent


def only_retain_last_1(input_list):
    result_list = []
    current_sequence = 0

    for element in input_list:
        if element == 1:
            current_sequence += 1
        else:
            current_sequence = 0

        result_list.append(1 if current_sequence == 1 else 0)

    return result_list


# 输入由0和1组成的list，返回所有1的位置
def getPosition(input_list):
    drift_pos = []
    for i, values in enumerate(input_list):
        if values == 1:
            drift_pos.append(i)
    return drift_pos


def draw_single_dataset(file, d, a):
    filename = os.path.basename(file)
    data = np.asarray(pd.read_csv(f'{root}/data/biobankdata/{file}_filled.csv', delimiter=",", header=None))
    data = data[1:]
    data = np.array(data)
    data = np.transpose(data)

    data1 = data[[1, 6], :]
    data1 = data1.astype(float)

    data2 = data[[2, 3, 4, 5], :]
    data2 = data2.astype(float)

    '''time = data[0, :]
    time = [timestamp.split()[0] + ' ' + timestamp.split()[1] for timestamp in time]
    input_format = "%Y-%m-%d %H:%M:%S.%f%z"
    output_format = "%Y-%m-%d %H:%M:%S"
    time = [datetime.strptime(date, input_format).strftime(output_format) for date in time]
    x_time = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in time]'''

    pkl1 = pd.read_pickle(
        f'{root}/results/exp_2023_ijcai/{file}_filled_binned/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_filled_binned_windowlength{d}_overlap{a}.pkl')
    content1 = list(pkl1.values())[0]

    HDDDM = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_filled_binned/full-1/HDDDM_bi_predictions_{file}_filled_binned.csv',
                        delimiter=",", header=None).iloc[:, 0].tolist()

    '''print(f'length of slidSHAPs predications: {len(content1)}')
    print(f'length of original time series: {len(data[0])}')
    print(f'drift position on the slidSHAPs axis: {getPosition(content1)}')'''

    # 通过插值将长度扩展为目标长度
    content1 = np.array(content1)
    new_indices = np.linspace(0, len(content1) - 1, len(data[0]), endpoint=True).astype(int)
    timeseries_predictions = content1[new_indices]
    '''print(f'drift position on the time series axis: {getPosition(timeseries_predictions)}')'''
    timeseries_predictions = only_retain_last_1(timeseries_predictions)
    # print(f'drift position on the time series axis (after reduce): {getPosition(timeseries_predictions)}')

    # data2 = [np.nan_to_num(row, nan=0) for row in data2]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(15, 5)
    fig.suptitle(f'{file}', fontsize=16, fontweight='bold', y=0.8)
    legend_labels1 = ['acc', 'MET']
    colors1 = ['lightblue', '#FF94A8']

    legend_labels2 = ['light', 'MVPA', 'sedentary', 'sleep']
    colors2 = ['lightblue', '#FFCCCC', '#FFDDBB', '#E6E6FA']

    for i, row in enumerate(data1):
        ax1.plot(row, label=legend_labels1[i], linewidth=0.3, color=colors1[i])
    for i, value in enumerate(timeseries_predictions):
        if value == 1:
            ax1.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=1)

    for i, value in enumerate(HDDDM):
        if value == 1:
            ax1.axvline(x=i, color='blue', linestyle='--', linewidth=0.8, alpha=1)
    ax1.set_xticks([])

    # 色块
    for i, row in enumerate(data2):
        # ax2.fill_between(x_time, 0, row.astype(float), label=legend_labels2[i], color=colors2[i], alpha=0.7)
        ax2.fill_between(np.arange(len(row)), 0, row.astype(float), label=legend_labels2[i], color=colors2[i], alpha=0.7)

    '''for i, row in enumerate(data2):
        ax2.plot(row, label=legend_labels2[i], linewidth=0.3, color=colors2[i])'''

    ax2.set_yticks([0, 1])

    existing_handles, existing_labels = plt.gca().get_legend_handles_labels()
    # 新图例的句柄和标签
    new_handles = [plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='slidSHAPs detection'),
                   plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=1, label='HDDDM detection'),
                   plt.Line2D([0], [0], color='green', linestyle='--', linewidth=1, label='ADWIN detection')]
    all_handles = existing_handles + new_handles
    all_labels = existing_labels + ['slidSHAPs detection', 'HDDDM detection', 'ADWIN detection']

    fig.legend(handles=all_handles, labels=all_labels, loc='upper left', bbox_to_anchor=(0.9, 0.73), prop={'size': 8})


    ax1.set_position([0.125, 0.415, 0.775, 0.3])  # [left, bottom, width, height]
    ax2.set_position([0.125, 0.3, 0.775, 0.075])

    # plt.show()
    plt.savefig(f'../data/biobankdata_plots/{filename}_filled.png')
    print(f'time series is saved in data/biobankdata_plots/{filename}_filled.png')
    plt.close()


def draw_accs(file, d, a):
    filename = os.path.basename(file)
    data = np.asarray(pd.read_csv(f'{root}/data/biobankdata/{file}_filled.csv', delimiter=",", header=None))
    data = data[1:]
    data = np.array(data)
    data = np.transpose(data)

    data1 = data[[1, 2, 3, 4, 5, 6, 7, 8, 9], :]
    data1 = data1.astype(float)

    '''
    # The first colomn of time in accs.csv, accs_after_filling.csv and accs_after_filling_and_binning.csv
    # should be "%Y-%m-%d %H:%M:%S.%f%z", although the %Y-%m-%ds are not true, but we need this format.
    time = data[0, :]
    time = [timestamp.split()[0] + ' ' + timestamp.split()[1] for timestamp in time]
    input_format = "%Y-%m-%d %H:%M:%S.%f%z"
    output_format = "%Y-%m-%d %H:%M:%S"
    time = [datetime.strptime(date, input_format).strftime(output_format) for date in time]
    x_time = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in time]
    '''

    pkl1 = pd.read_pickle(
        f'{root}/results/exp_2023_ijcai/{file}_filled_binned/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_filled_binned_windowlength{d}_overlap{a}.pkl')

    content1 = list(pkl1.values())[0]

    # 通过插值将长度扩展为目标长度
    content1 = np.array(content1)
    new_indices = np.linspace(0, len(content1) - 1, len(data[0]), endpoint=True).astype(int)
    timeseries_predictions = content1[new_indices]
    '''print(f'drift position on the time series axis: {getPosition(timeseries_predictions)}')'''

    timeseries_predictions = only_retain_last_1(timeseries_predictions)
    # print(f'drift position on the time series axis: {getPosition(timeseries_predictions)}')

    plt.figure(figsize=(20, 2))
    plt.suptitle(f'{filename}_filled', fontsize=16, fontweight='bold')
    header = ('acc1', 'acc2', 'acc3', 'acc4', 'acc5', 'acc6', 'acc7', 'acc8', 'acc9')

    for i, row in enumerate(data1):
        # plt.plot(x_time, row, label=header[i], linewidth=0.2, alpha=0.5)
        plt.plot(row, label=header[i], linewidth=0.2, alpha=0.5)
    for i, value in enumerate(timeseries_predictions):
        if value == 1:
            # plt.axvline(x=x_time[i], color='red', linestyle='-', linewidth=0.8, alpha=0.01)
            plt.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=1)

    HDDDM = pd.read_csv(
        f'{root}/results/exp_2023_ijcai/{file}_filled_binned/full-1/HDDDM_bi_predictions_{file}_filled_binned.csv',
        delimiter=",", header=None).iloc[:, 0].tolist()
    for i, value in enumerate(HDDDM):
        if value == 1:
            plt.axvline(x=i, color='blue', linestyle='--', linewidth=0.8, alpha=2)
    #plt.set_xticks([])

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
    existing_handles, existing_labels = plt.gca().get_legend_handles_labels()
    # 新图例的句柄和标签
    new_handles = [plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='slidSHAPs detection'),
                   plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=1, label='HDDDM detection'),
                   plt.Line2D([0], [0], color='green', linestyle='--', linewidth=1, label='ADWIN detection')]
    all_handles = existing_handles + new_handles
    all_labels = existing_labels + ['slidSHAPs detection', 'HDDDM detection', 'ADWIN detection']

    plt.legend(handles=all_handles, labels=all_labels, loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 8})



    # plt.show()
    plt.savefig(f'../data/biobankdata_plots/{filename}_filled.png')
    print(f'time series is saved in data/biobankdata_plots/{filename}_filled.png')
    plt.close()


if __name__ == '__main__':
    #draw_single_dataset('dataset9-timeSeries', 200, 140)
    draw_accs('accs', 200, 140)
