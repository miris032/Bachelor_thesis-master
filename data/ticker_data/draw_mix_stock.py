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

    data = np.asarray(pd.read_csv(f'{root}/ticker_data/' + f'{file}' + '.csv', delimiter=",", header=None))
    data = np.transpose(data)
    time = data[0, 1:]
    # data = data[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 1:]
    data = data[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1:]
    data = data.astype(float)
    input_format = "%Y-%m-%d %H:%M:%S%z"
    output_format = "%Y-%m-%d %H:%M:%S%z"
    time = [datetime.strptime(date, input_format).strftime(output_format) for date in time]
    x_time = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S%z') for d in time]

    pkl1 = pd.read_pickle(
        f'{root}/../results/exp_2023_ijcai/' + f'{file}_binned50' + '/full-1/slidshap_drifts_bi_predictions_Realworld_' + f'{file}_binned50' + '_windowlength' + f'{d}' + '_overlap' + f'{a}' + '.pkl')
    content1 = list(pkl1.values())[0]

    # 通过插值将长度扩展为目标长度
    content1 = np.array(content1)
    new_indices = np.linspace(0, len(content1) - 1, len(data[0]), endpoint=True).astype(int)
    timeseries_predictions = content1[new_indices]
    print(f'drift position on the time series axis: {getPosition(timeseries_predictions)}')
    # timeseries_predictions = only_retain_last_1(timeseries_predictions)
    # print(f'drift position on the time series axis (after reduce): {getPosition(timeseries_predictions)}')

    plt.figure(figsize=(20, 5))
    header = (f'1COV.DE_{filename}', f'ADS.DE_{filename}', f'AIR.DE_{filename}', f'ALV.DE_{filename}', f'BAS.DE_{filename}', f'BMW.DE_{filename}', f'CBK.DE_{filename}', f'DTE.DE_{filename}', f'EOAN.DE_{filename}', f'HEI.DE_{filename}',
              f'HNR1.DE_{filename}', f'IFX.DE_{filename}', f'MRK.DE_{filename}', f'RHM.DE_{filename}', f'RWE.DE_{filename}', f'SAP.DE_{filename}', f'SIE.DE_{filename}', f'SY1.DE_{filename}', f'VNA.DE_{filename}')
    colors = ['#FF9933', '#FF6666', '#66FF66', '#6666FF', '#996699', '#66CCCC']

    for i, row in enumerate(data):
        # plt.plot(x_time2, row, label=header[i], linewidth=0.3)
        plt.plot(x_time, row, label=header[i], linewidth=0.3)
    '''for i, value in enumerate(timeseries_predictions):
        if value == 1:
            plt.axvline(x=x_time[i], color='red', linestyle='-', linewidth=0.8, alpha=0.05)'''
            # plt.axvline(x=i, color='red', linestyle='-', linewidth=0.8, alpha=0.05)

    plt.suptitle(f'{filename}' + '', fontsize=16, fontweight='bold')
    plt.subplots_adjust(hspace=0.8)
    plt.legend(loc=2, bbox_to_anchor=(1, 1), prop={'size': 6})
    # plt.legend(loc=4)

    plt.show()
    # plt.savefig('../data/ticker_data_result/' + f'{filename}.png')
    # plt.close()


if __name__ == '__main__':
    draw_plots('Open_hourly', 50, 35)
