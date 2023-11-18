import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


root = Path(__file__).resolve().parent.parent


def load_data0(data):

    path = f'{root}/' + data

    if os.path.exists(path + '.pkl'):
        _mydata = pd.read_pickle(path + '.pkl')

    elif os.path.exists(path + '.txt'):
        _mydata = np.loadtxt(path + '.txt', delimiter=",")

    elif os.path.exists(path + '.csv'):
        _mydata = np.asarray(pd.read_csv(path + '.csv', delimiter=","))
    #         _mydata = _mydata)
    else:
        raise FileNotFoundError(path+'.*')

    # _mydata = np.delete(_mydata, [1, 2, 3, 4], axis=1)
    _mydata = np.array(_mydata)
    _mydata = np.transpose(_mydata)


    return _mydata


def drawPlot(data):

    data1 = np.transpose(data)[:, [1, 6]]
    data1 = np.transpose(data1)
    data2 = np.transpose(data)[:, [2, 3, 4, 5]]
    data2 = np.transpose(data2)
    # data2 = [np.nan_to_num(row, nan=0) for row in data2]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(15, 5)
    fig.suptitle('dataset1-timeSeries_new', fontsize=16, fontweight='bold')
    legend_labels1 = ['acc', 'MET']
    colors1 = ['lightblue', 'red']

    legend_labels2 = ['light', 'MVPA', 'sedentary', 'sleep']
    colors2 = ['lightblue', '#FFCCCC', '#FFDDBB', '#E6E6FA']

    for i, row in enumerate(data1):
        ax1.plot(row, label=legend_labels1[i], linewidth=0.3, color=colors1[i])

    ax1.legend(bbox_to_anchor=(1, 1), loc='upper left')


    for i, row in enumerate(data2):
        ax2.fill_between(range(len(data2[i])), 0, data2[i], label=legend_labels2[i], color=colors2[i], alpha=0.7)

    '''for i, row in enumerate(data2):
        ax2.plot(row, label=legend_labels2[i], linewidth=0.3, color=colors2[i])'''

    ax2.set_yticks([0, 1])
    ax2.set_position([0.125, 0.2, 0.775, 0.15])  # [left, bottom, width, height]

    fig.suptitle('dataset1-timeSeries_new', fontsize=16, fontweight='bold')

    ax2.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()





if __name__ == '__main__':
    drawPlot(load_data0('biobankdata/dataset9-timeSeries'))
    #print(load_data0('biobankdata/test'))