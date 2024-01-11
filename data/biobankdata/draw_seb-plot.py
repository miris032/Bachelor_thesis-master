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


def drawPlot(file, d, a):
    data = np.asarray(pd.read_csv(f'{file}' + '.csv', delimiter=",", header=None))
    data = data[1:]
    data = np.array(data)
    data = np.transpose(data)

    data1 = data[[1, 6], :]
    data1 = data1.astype(float)

    data2 = data[[2, 3, 4, 5], :]
    data2 = data2.astype(float)

    time = data[0, :]
    time = [timestamp.split()[0] + ' ' + timestamp.split()[1] for timestamp in time]
    input_format = "%Y-%m-%d %H:%M:%S.%f%z"
    output_format = "%Y-%m-%d %H:%M:%S"
    time = [datetime.strptime(date, input_format).strftime(output_format) for date in time]
    x_time = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in time]

    pkl1 = pd.read_pickle(f'{root}/../results/exp_2023_ijcai/' + f'{file}' + '/full-1/slidshap_drifts_bi_predictions_Realworld_' + f'{file}' + '_windowlength' + f'{d}' + '_overlap' + f'{a}' + '.pkl')
    content1 = list(pkl1.values())[0]



    print(f'slidSHAPs predications: {content1}')
    print(f'length of slidSHAPs predications: {len(content1)}')
    print(f'length of original time series: {len(data[0])}')
    print(f'drift position on the slidSHAPs axis: {getPosition(content1)}')


    # 通过插值将长度扩展为目标长度
    content1 = np.array(content1)
    new_indices = np.linspace(0, len(content1) - 1, len(data[0]), endpoint=True).astype(int)
    timeseries_predictions = content1[new_indices]
    print(f'drift position on the time series axis: {getPosition(timeseries_predictions)}')
    timeseries_predictions = only_retain_last_1(timeseries_predictions)
    print(f'drift position on the time series axis (after reduce): {getPosition(timeseries_predictions)}')


    # data2 = [np.nan_to_num(row, nan=0) for row in data2]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(15, 5)
    fig.suptitle(f'{file}', fontsize=16, fontweight='bold')
    legend_labels1 = ['acc', 'MET']
    colors1 = ['lightblue', '#FF94A8']

    legend_labels2 = ['light', 'MVPA', 'sedentary', 'sleep']
    colors2 = ['lightblue', '#FFCCCC', '#FFDDBB', '#E6E6FA']

    for i, row in enumerate(data1):
        ax1.plot(row, label=legend_labels1[i], linewidth=0.3, color=colors1[i])
    for i, value in enumerate(timeseries_predictions):
        if value == 1:
            ax1.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax1.legend(bbox_to_anchor=(1, 1), loc='upper left')


    for i, row in enumerate(data2):
        ax2.fill_between(x_time, 0, row.astype(float), label=legend_labels2[i], color=colors2[i], alpha=0.7)

    '''for i, row in enumerate(data2):
        ax2.plot(row, label=legend_labels2[i], linewidth=0.3, color=colors2[i])'''

    ax2.set_yticks([0, 1])
    ax2.legend(bbox_to_anchor=(1, 1), loc='upper left')

    ax1.set_position([0.125, 0.415, 0.775, 0.3])  # [left, bottom, width, height]
    ax2.set_position([0.125, 0.3, 0.775, 0.075])

    plt.show()





if __name__ == '__main__':
    drawPlot('dataset1-timeSeries_after_filling', 200, 140)
