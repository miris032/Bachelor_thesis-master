import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def getHeader(file):
    with open(file, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # 读取第一行作为列名
    return header




root = Path(__file__).resolve().parent.parent
def load_data0(data):

    path = f'{root}/data/' + data

    if os.path.exists(path + '.pkl'):
        _mydata = pd.read_pickle(path + '.pkl')

    elif os.path.exists(path + '.txt'):
        _mydata = np.loadtxt(path + '.txt', delimiter=",")

    elif os.path.exists(path + '.csv'):
        _mydata = np.asarray(pd.read_csv(path + '.csv', delimiter=","))
    #         _mydata = _mydata)
    else:
        raise FileNotFoundError(path+'.*')

    return _mydata




def drawPlot(data):

    '''max_length = max(len(row) for row in data)
    point_spacing = 0.8
    fig_width = max_length * point_spacing
    fig_height = 6

    plt.figure(figsize=(fig_width, fig_height))
    for i, row in enumerate(data):
        plt.plot(row, label=f'Data Group {i + 1}')

    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper left')
    plt.show()'''



    plt.figure()
    for i, row in enumerate(data):
        plt.plot(row, label=f'Data Group {i + 1}', linewidth=0.3)
    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper left')
    plt.title('_window_width(d): 1000, _overlap(a): 900')
    plt.show()




def draw_4_plots(file):

    directory_path = '/home/miris/Desktop/Bachelor_thesis-master/data/ticker_data/columns_minutely'
    '''with os.scandir(directory_path) as entries:
        for entry in entries:
            if entry.is_dir():
                print(entry.name)'''


    filename = os.path.basename(file)
    data1 = load_data0('ticker_data_result/columns_minutely/' + filename + '/50, 30%')
    data2 = load_data0('ticker_data_result/columns_minutely/' + filename + '/50, 70%')
    data3 = load_data0('ticker_data_result/columns_minutely/' + filename + '/200, 30%')
    data4 = load_data0('ticker_data_result/columns_minutely/' + filename + '/200, 70%')

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    #header = ('Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume')
    # header = ('Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock', 'Splits')
    header = ('COV.DE_Open', 'ADS.DE_Open', 'AIR.DE_Open', 'ALV.DE_Open', 'BAS.DE_Open', 'BMW.DE_Open',
              'CBK.DE_Open', 'DTE_Open', 'EOAN.DE_Open', 'HEI.DE_Open')


    for i, row in enumerate(data1):
        ax1.plot(row, label=header[i], linewidth=0.3)
    ax1.set_title('length d = 50, overlap a = 30%')

    for i, row in enumerate(data2):
        ax2.plot(row, label=header[i], linewidth=0.3)
    ax2.set_title('length d = 50, overlap a = 70%')

    for i, row in enumerate(data3):
        ax3.plot(row, label=header[i], linewidth=0.3)
    ax3.set_title('length d = 200, overlap a = 30%')

    for i, row in enumerate(data4):
        ax4.plot(row, label=header[i], linewidth=0.3)
    ax4.set_title('length d = 200, overlap a = 70%')

    # plt.legend(bbox_to_anchor=(0.1, 0.1), loc='upper left')
    fig.set_size_inches(20, 8)
    fig.suptitle(f'{filename}', fontsize=16, fontweight='bold')
    plt.subplots_adjust(hspace=0.8)
    plt.legend(loc=2, bbox_to_anchor=(1.005, 1.6))

    plt.savefig('../data/ticker_data_result/columns_minutely/plots/' + f'{filename}.png')
    # plt.show()
    plt.close()






if __name__ == '__main__':

    '''
    data = load_data("results/_window_width(d): 1000, _overlap(a): 900")
    drawPlot(data)
    '''

    #draw_4_plots('Adj Close')
    '''draw_4_plots('Close')
    draw_4_plots('High')
    draw_4_plots('Low')
    draw_4_plots('Open')
    draw_4_plots('Volume')'''

    data1 = load_data0('ticker_data_result/columns_minutely/Volume/50, 30%')
    data2 = load_data0('ticker_data_result/columns_minutely/Volume/50, 70%')
    data3 = load_data0('ticker_data_result/columns_minutely/Volume/200, 30%')
    data4 = load_data0('ticker_data_result/columns_minutely/Volume/200, 70%')

    print(data1.shape)
    print(data2.shape)
    print(data3.shape)
    print(data4.shape)



