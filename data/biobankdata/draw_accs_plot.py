import csv
import os
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

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


def drawPlot(file, d, a):
    filename = os.path.basename(file)
    data = np.asarray(pd.read_csv(f'{file}' + '.csv', delimiter=",", header=None))
    data = data[1:]
    data = np.array(data)
    data = np.transpose(data)

    data1 = data[[1, 2, 3, 4, 5, 6, 7, 8, 9], :]
    data1 = data1.astype(float)

    time = data[0, :]
    time = [timestamp.split()[0] + ' ' + timestamp.split()[1] for timestamp in time]

    # The first colomn of time in accs.csv, accs_after_filling.csv and accs_after_filling_and_binning.csv
    # should be "%Y-%m-%d %H:%M:%S.%f%z", although the %Y-%m-%ds are not true, but we need this format.

    input_format = "%Y-%m-%d %H:%M:%S.%f%z"
    output_format = "%Y-%m-%d %H:%M:%S"
    time = [datetime.strptime(date, input_format).strftime(output_format) for date in time]
    x_time = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in time]

    pkl1 = pd.read_pickle(
        f'{root}/../results/exp_2023_ijcai/' + f'{file}' + '' + '/full-1/slidshap_drifts_bi_predictions_Realworld_' + f'{file}' + '' + '_windowlength' + f'{d}' + '_overlap' + f'{a}' + '.pkl')
    content1 = list(pkl1.values())[0]
    print(f'slidSHAPs predications: {content1}')
    print(f'length of slidSHAPs predications: {len(content1)}')
    print(f'length of original time series: {len(data[0])}')

    # 通过插值将长度扩展为目标长度
    content1 = np.array(content1)
    new_indices = np.linspace(0, len(content1) - 1, len(data[0]), endpoint=True).astype(int)
    timeseries_predictions = content1[new_indices]
    print(f'drift position on the time series axis: {getPosition(timeseries_predictions)}')
    # timeseries_predictions = only_retain_last_1(timeseries_predictions)
    # print(f'drift position on the time series axis: {getPosition(timeseries_predictions)}')

    plt.figure(figsize=(15, 3))
    plt.suptitle(f'{filename}', fontsize=16, fontweight='bold')
    header = ('acc1', 'acc2', 'acc3', 'acc4', 'acc5', 'acc6', 'acc7', 'acc8', 'acc9')

    for i, row in enumerate(data1):
        # plt.plot(x_time, row, label=header[i], linewidth=0.2, alpha=0.5)
        plt.plot(row, label=header[i], linewidth=0.2, alpha=0.5)
    for i, value in enumerate(timeseries_predictions):
        if value == 1:
            # plt.axvline(x=x_time[i], color='red', linestyle='-', linewidth=0.8, alpha=0.01)
            plt.axvline(x=i, color='red', linestyle='-', linewidth=0.8, alpha=0.01)

    HDDDM = [1199, 3199, 3999, 5599, 6399, 7199, 7999, 9599, 10399, 13999, 14799, 16399, 17199, 18399, 19199, 20399, 21199]
    for pos in HDDDM:
        plt.axvline(x=pos, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

    plt.legend(bbox_to_anchor=(1, 1.03), loc='upper left')

    plt.show()
    # plt.savefig('../biobankdata_plots/accs_after_filling/' + f'{filename}.png')
    # plt.close()


if __name__ == '__main__':
    # draw accs_after_filling, not accs_after_filling_and_binning as the original ts
    drawPlot('accs_after_filling_events', 200, 140)
