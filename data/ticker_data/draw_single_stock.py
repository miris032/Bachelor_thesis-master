import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
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


def draw_plots(file, d, a):
    filename = os.path.basename(file)

    data1 = np.asarray(pd.read_csv(f'{root}/ticker_data/' + f'{file}' + '.csv', delimiter=",", header=None))
    data1 = np.transpose(data1)
    time1 = data1[0, 1:]
    data1 = data1[[1, 2, 3, 4], 1:]
    data1 = data1.astype(float)
    # input_format = "%Y-%m-%d"
    # output_format = "%Y-%m-%d"
    input_format = "%Y-%m-%d %H:%M:%S%z"
    output_format = "%Y-%m-%d %H:%M:%S%z"
    time1 = [datetime.strptime(date, input_format).strftime(output_format) for date in time1]
    # x_time1 = [datetime.strptime(d, "%Y-%m-%d") for d in time1]
    x_time1 = [datetime.strptime(d, "%Y-%m-%d %H:%M:%S%z") for d in time1]

    pkl1 = pd.read_pickle(
        f'{root}/../results/exp_2023_ijcai/' + f'{file}_binned50' + '/full-1/slidshap_drifts_bi_predictions_Realworld_' + f'{file}_binned50' + '_windowlength' + f'{d}' + '_overlap' + f'{a}' + '.pkl')
    content1 = list(pkl1.values())[0]

    # 通过插值将长度扩展为目标长度
    content1 = np.array(content1)
    new_indices = np.linspace(0, len(content1) - 1, len(data1[0]), endpoint=True).astype(int)
    timeseries_predictions = content1[new_indices]
    print(f'drift position on the time series axis: {getPosition(timeseries_predictions)}')
    # timeseries_predictions = only_retain_last_1(timeseries_predictions)
    # print(f'drift position on the time series axis (after reduce): {getPosition(timeseries_predictions)}')

    fig, (ax1) = plt.subplots(1, 1)
    header = ('Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume')
    colors = ['#FF6666', '#66FF66', '#FF9933', '#6666FF', '#66CCCC', '#996699']

    for i, row in enumerate(data1):
        ax1.plot(x_time1, row, label=header[i], linewidth=0.3, alpha=0.5)
        # ax1.plot(row, label=header[i], linewidth=0.3, alpha=0.5)
    for i, value in enumerate(timeseries_predictions):
        if value == 1:
            # ax1.axvline(x=x_time1[i], color='red', linestyle='-', linewidth=0.8, alpha=0.05)
            ax1.axvline(x=x_time1[i], color='red', linestyle='-', linewidth=0.8, alpha=0.05)

    HDDDM = [299, 799, 1299, 1599, 1799, 2699]
    for pos in HDDDM:
        ax1.axvline(x=x_time1[pos], color='blue', linestyle='--', linewidth=0.8, alpha=0.5)

    #ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    fig.set_size_inches(20, 5)
    fig.suptitle(f'{filename}' + '', fontsize=16, fontweight='bold')
    plt.subplots_adjust(hspace=0.8)
    plt.legend(loc=2, bbox_to_anchor=(1.005, 1))

    plt.show()
    # plt.savefig('../data/ticker_data_result/' + f'{filename}.png')
    # plt.close()


if __name__ == '__main__':
    # draw_plots('daily/ADS.DE')
    draw_plots('BMW.DE_hourly', 200, 140)
