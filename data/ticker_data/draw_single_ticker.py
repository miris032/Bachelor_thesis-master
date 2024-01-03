import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime



root = Path(__file__).resolve().parent.parent




def draw_plots(file):
    filename = os.path.basename(file)

    data1 = np.asarray(pd.read_csv(f'{root}/ticker_data/' + f'{file}' + '.csv', delimiter=",", header=None))
    data1 = np.transpose(data1)
    time1 = data1[0, 1:]
    data1 = data1[[1, 2, 3, 4], 1:]
    data1 = data1.astype(float)
    input_format = "%Y-%m-%d %H:%M:%S%z"
    output_format = "%Y-%m-%d %H:%M:%S%z"
    time1 = [datetime.strptime(date, input_format).strftime(output_format) for date in time1]
    x_time1 = [datetime.strptime(d, "%Y-%m-%d %H:%M:%S%z") for d in time1]

    '''data2 = np.asarray(pd.read_csv(f'{root}/ticker_data/hourly/' + f'{file}' + '.csv', delimiter=",", header=None))
    data2 = np.transpose(data2)
    time2 = data2[1, 1:]
    data2 = data2[[2, 3, 4, 5], 1:]
    data2 = data2.astype(float)
    input_format = "%Y-%m-%d %H:%M:%S%z"
    output_format = "%Y-%m-%d %H:%M:%S%z"
    time2 = [datetime.strptime(date, input_format).strftime(output_format) for date in time2]
    x_time2 = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S%z') for d in time2]

    data3 = np.asarray(pd.read_csv(f'{root}/ticker_data/minutely/' + f'{file}' + '.csv', delimiter=",", header=None))
    data3 = np.transpose(data3)
    time3 = data3[1, 1:]
    data3 = data3[[2, 3, 4, 5], 1:]
    data3 = data3.astype(float)
    input_format = "%Y-%m-%d %H:%M:%S%z"
    output_format = "%Y-%m-%d %H:%M:%S%z"
    time3 = [datetime.strptime(date, input_format).strftime(output_format) for date in time3]
    x_time3 = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S%z') for d in time3]'''

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    header = ('Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume')
    colors = ['#FF9933', '#FF6666', '#66FF66', '#6666FF', '#996699', '#66CCCC']

    for i, row in enumerate(data1):
        ax1.plot(x_time1, row, label=header[i], linewidth=0.3)
        # ax1.plot(row, label=header[i], linewidth=0.3)

    '''for i, row in enumerate(data2):
        ax2.plot(x_time2, row, label=header[i], linewidth=0.3)

    for i, row in enumerate(data3):
        ax3.plot(row, label=header[i], linewidth=0.3, color=colors[i])'''

    fig.set_size_inches(20, 5)
    fig.suptitle(f'{filename}' + '_', fontsize=16, fontweight='bold')
    plt.subplots_adjust(hspace=0.8)
    plt.legend(loc=2, bbox_to_anchor=(1.005, 1.6))

    plt.show()
    # plt.savefig('../data/ticker_data_result/' + f'{filename}.png')
    # plt.close()


if __name__ == '__main__':
    draw_plots('1COV.DE_minutely_oneweek')
