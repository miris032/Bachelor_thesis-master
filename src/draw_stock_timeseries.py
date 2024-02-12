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


def draw_single_stock(file, d, a):

    time_category = file.split('_')[1]
    file_name = file.split('_')[0]

    data1 = np.asarray(pd.read_csv(f'{root}/data/ticker_data/{time_category}/{file_name}.csv', delimiter=",", header=None))
    data1 = np.transpose(data1)
    # time1 = data1[0, 1:]
    data1 = data1[[1, 2, 3, 4], 1:]
    data1 = data1.astype(float)

    '''if time_category == 'daily':
        input_format = "%Y-%m-%d"
        output_format = "%Y-%m-%d"
    else:
        input_format = "%Y-%m-%d %H:%M:%S%z"
        output_format = "%Y-%m-%d %H:%M:%S%z"
    time1 = [datetime.strptime(date, input_format).strftime(output_format) for date in time1]
    if time_category == 'daily':
        x_time1 = [datetime.strptime(d, "%Y-%m-%d") for d in time1]
    else:
        x_time1 = [datetime.strptime(d, "%Y-%m-%d %H:%M:%S%z") for d in time1]'''

    pkl1 = pd.read_pickle(
        f'{root}/results/exp_2023_ijcai/{file}_binned/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_binned_windowlength{d}_overlap{a}.pkl')
    content1 = list(pkl1.values())[0]

    HDDDM = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_binned/full-1/HDDDM_bi_predictions_{file}.csv',
                        delimiter=",", header=None).iloc[:, 0].tolist()
    ADWIN = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_binned/full-1/ADWIN_bi_predictions_{file}.csv',
                        delimiter=",", header=None).iloc[:, 0].tolist()

    # 通过插值将长度扩展为目标长度
    content1 = np.array(content1)
    new_indices = np.linspace(0, len(content1) - 1, len(data1[0]), endpoint=True).astype(int)
    timeseries_predictions = content1[new_indices]
    timeseries_predictions = only_retain_last_1(timeseries_predictions)

    plt.figure(figsize=(20, 3))
    header = ('Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume')
    colors = ['#FF6666', '#66FF66', '#FF9933', '#6666FF', '#66CCCC', '#996699']

    for i, row in enumerate(data1):
        # ax1.plot(x_time1, row, label=header[i], linewidth=0.3, alpha=0.5)
        plt.plot(row, label=header[i], linewidth=0.3, alpha=0.5)
    for i, value in enumerate(timeseries_predictions):
        if value == 1:
            # ax1.axvline(x=x_time1[i], color='red', linestyle='-', linewidth=0.8, alpha=0.05)
            plt.axvline(x=i, color='red', linestyle='-', linewidth=0.8, alpha=1)
    for i, value in enumerate(HDDDM):
        if value == 1:
            # ax1.axvline(x=x_time1[i], color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
            plt.axvline(x=i, color='blue', linestyle='--', linewidth=0.8, alpha=1)
    for i, value in enumerate(ADWIN):
        if value == 1:
            plt.axvline(x=i, color='green', linestyle='--', linewidth=0.8, alpha=1)

    #ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    existing_handles, existing_labels = plt.gca().get_legend_handles_labels()
    # 新图例的句柄和标签
    new_handles = [plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='slidSHAPs detection'),
                   plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=1, label='HDDDM detection'),
                   plt.Line2D([0], [0], color='green', linestyle='--', linewidth=1, label='ADWIN detection')]
    all_handles = existing_handles + new_handles
    all_labels = existing_labels + ['slidSHAPs detection', 'HDDDM detection', 'ADWIN detection']

    plt.legend(handles=all_handles, labels=all_labels, loc='upper left', bbox_to_anchor=(1.005, 1), prop={'size': 8})

    plt.suptitle(f'{file}' + '', fontsize=16, fontweight='bold')
    plt.subplots_adjust(hspace=0.8)
    # plt.legend(loc=2, bbox_to_anchor=(1.005, 1))

    # plt.show()
    plt.savefig(f'../data/ticker_data_plots/{file}.png')
    print(f'time series is saved in data/biobankdata_plots/{file}.png')
    plt.close()


def draw_mix_stock(file, d, a):
    filename = os.path.basename(file)

    data = np.asarray(pd.read_csv(f'{root}/data/ticker_data/{file}.csv', delimiter=",", header=None))
    data = np.transpose(data)
    '''time = data[0, 1:]'''
    data = data[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1:]
    data = data.astype(float)
    '''input_format = "%Y-%m-%d %H:%M:%S%z"
    output_format = "%Y-%m-%d %H:%M:%S%z"
    time = [datetime.strptime(date, input_format).strftime(output_format) for date in time]
    x_time = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S%z') for d in time]'''

    pkl1 = pd.read_pickle(
        f'{root}/results/exp_2023_ijcai/{file}_binned/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_binned_windowlength{d}_overlap{a}.pkl')
    content1 = list(pkl1.values())[0]

    HDDDM = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_binned/full-1/HDDDM_bi_predictions_{file}.csv',
                        delimiter=",", header=None).iloc[:, 0].tolist()
    ADWIN = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_binned/full-1/ADWIN_bi_predictions_{file}.csv',
                        delimiter=",", header=None).iloc[:, 0].tolist()

    # 通过插值将长度扩展为目标长度
    content1 = np.array(content1)
    new_indices = np.linspace(0, len(content1) - 1, len(data[0]), endpoint=True).astype(int)
    timeseries_predictions = content1[new_indices]
    timeseries_predictions = only_retain_last_1(timeseries_predictions)

    plt.figure(figsize=(20, 3))
    header = (f'1COV.DE', f'ADS.DE', f'AIR.DE', f'ALV.DE', f'BAS.DE', f'BMW.DE', f'CBK.DE', f'DTE.DE', f'EOAN.DE', f'HEI.DE',
              f'HNR1.DE', f'IFX.DE', f'MRK.DE', f'RHM.DE', f'RWE.DE', f'SAP.DE', f'SIE.DE', f'SY1.DE', f'VNA.DE')
    colors = ['#FF9933', '#FF6666', '#66FF66', '#6666FF', '#996699', '#66CCCC']

    for i, row in enumerate(data):
        # plt.plot(x_time2, row, label=header[i], linewidth=0.3)
        # plt.plot(x_time, row, label=header[i], linewidth=0.3)
        plt.plot(row, label=header[i], linewidth=0.3)

    legend_handles = []
    for i, value in enumerate(timeseries_predictions):
        if value == 1:
            # x=x_time[i]
            plt.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=1)
    for i, value in enumerate(HDDDM):
        if value == 1:
            plt.axvline(x=i, color='blue', linestyle='--', linewidth=0.8, alpha=1)
    for i, value in enumerate(ADWIN):
        if value == 1:
            plt.axvline(x=i, color='green', linestyle='--', linewidth=0.8, alpha=1)
    insert_starts = (1001, 2501, 4001, 5501, 7001)
    insert_ends = (1500, 3000, 4500, 6000, 7500)
    ylim = plt.gca().get_ylim()
    for j in range(len(insert_starts)):
        print(f'{insert_starts[j]}-{insert_ends[j]}')
        plt.fill_between(np.linspace(insert_starts[j], insert_ends[j]), ylim[0], ylim[1], color='gray', hatch='///',
                         alpha=0.2)
    plt.suptitle(f'{filename}' + '', fontsize=16, fontweight='bold')
    plt.subplots_adjust(hspace=0.8)

    existing_handles, existing_labels = plt.gca().get_legend_handles_labels()
    # 新图例的句柄和标签
    new_handles = [plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='slidSHAPs detection'),
                   plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=1, label='HDDDM detection'),
                   plt.Line2D([0], [0], color='green', linestyle='--', linewidth=1, label='ADWIN detection')]
    all_handles = existing_handles + new_handles
    all_labels = existing_labels + ['slidSHAPs detection', 'HDDDM detection', 'ADWIN detection']

    plt.legend(handles=all_handles, labels=all_labels, loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 8})
    # plt.legend(loc=4)

    # plt.show()
    plt.savefig(f'../data/ticker_data_plots/{file}.png')
    print(f'time series is saved in data/biobankdata_plots/{file}.png')
    plt.close()
