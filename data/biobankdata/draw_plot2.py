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
        _mydata = np.asarray(pd.read_csv(path + '.csv', delimiter=",", header=None))
    #         _mydata = _mydata)
    else:
        raise FileNotFoundError(path+'.*')

    # _mydata = np.delete(_mydata, [1, 2, 3, 4], axis=1)
    _mydata = np.array(_mydata)
    _mydata = np.transpose(_mydata)

    return _mydata


def drawPlot(data):
    data1 = data[[0, 1], :]
    data2 = data[[2, 3, 4], :]
    data3 = data[[5, 6, 7], :]
    data4 = data[[8, 9, 10], :]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    fig.set_size_inches(16, 9)
    fig.suptitle('all', fontsize=16, fontweight='bold')

    categories1 = ['avg', 'sd']
    categories2 = ['x', 'y', 'z']

    x_labels = ['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', 'dataset6', 'dataset7', 'dataset8', 'dataset9', ]
    label_length = len(x_labels)

    for i, row in enumerate(data1):
        ax1.plot_timeseries(row, label=categories1[i], linewidth=1)
    ax1.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax1.set_xticks(np.arange(label_length))
    ax1.set_xticklabels([])
    ax1.set_title('overall Average & Standard deviation')

    for i, row in enumerate(data2):
        ax2.plot_timeseries(row, label=categories2[i], linewidth=1)
    ax2.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax2.set_xticks(np.arange(label_length))
    ax2.set_xticklabels([])
    ax2.set_title('calibration-Offset(g) on x, y, z axis')

    for i, row in enumerate(data3):
        ax3.plot_timeseries(row, label=categories2[i], linewidth=1)
    ax3.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax3.set_xticks(np.arange(label_length))
    ax3.set_xticklabels([])
    ax3.set_title('calibration-Slope on x, y, z axis')

    for i, row in enumerate(data4):
        ax4.plot_timeseries(row, label=categories2[i], linewidth=1)
    ax4.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax4.set_xticks(np.arange(label_length))
    ax4.set_xticklabels(x_labels)
    ax4.set_title('calibration-SlopeTemp on x, y, z axis')

    plt.subplots_adjust(hspace=0.3)

    #fig.suptitle('all', fontsize=16, fontweight='bold')
    plt.show()





if __name__ == '__main__':
    drawPlot(load_data0('biobankdata/all'))
    #print(load_data0('biobankdata/noise_test'))