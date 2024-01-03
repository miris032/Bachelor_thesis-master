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
    data1 = np.asarray(pd.read_csv(f'{root}/results/exp_2023_ijcai/' + f'{file}' + '/full-1/slidshaps_Realworld_' + f'{file}' + '_windowlength50_overlap15.txt', delimiter=",", header=None))
    data2 = np.asarray(pd.read_csv(f'{root}/results/exp_2023_ijcai/' + f'{file}' + '/full-1/slidshaps_Realworld_' + f'{file}' + '_windowlength50_overlap35.txt', delimiter=",", header=None))
    data3 = np.asarray(pd.read_csv(f'{root}/results/exp_2023_ijcai/' + f'{file}' + '/full-1/slidshaps_Realworld_' + f'{file}' + '_windowlength200_overlap60.txt', delimiter=",", header=None))
    data4 = np.asarray(pd.read_csv(f'{root}/results/exp_2023_ijcai/' + f'{file}' + '/full-1/slidshaps_Realworld_' + f'{file}' + '_windowlength200_overlap140.txt', delimiter=",", header=None))

    pkl1 = pd.read_pickle(f'{root}/results/exp_2023_ijcai/' + f'{file}' + '/full-1/slidshap_drifts_bi_predictions_Realworld_' + f'{file}' + '_windowlength50_overlap15.pkl')
    pkl2 = pd.read_pickle(f'{root}/results/exp_2023_ijcai/' + f'{file}' + '/full-1/slidshap_drifts_bi_predictions_Realworld_' + f'{file}' + '_windowlength50_overlap35.pkl')
    pkl3 = pd.read_pickle(f'{root}/results/exp_2023_ijcai/' + f'{file}' + '/full-1/slidshap_drifts_bi_predictions_Realworld_' + f'{file}' + '_windowlength200_overlap60.pkl')
    pkl4 = pd.read_pickle(f'{root}/results/exp_2023_ijcai/' + f'{file}' + '/full-1/slidshap_drifts_bi_predictions_Realworld_' + f'{file}' + '_windowlength200_overlap140.pkl')
    content1 = list(pkl1.values())[0]
    content2 = list(pkl2.values())[0]
    content3 = list(pkl3.values())[0]
    content4 = list(pkl4.values())[0]
    print(len(content1))
    print(len(content2))
    print(len(content3))
    print(len(content4))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    header = ('sin1', 'ADD', 'ADD', 'ADD', 'ADD', 'ADD')
    colors = ['#FF6666', '#6666FF', '#66FF66', '#FF9933', '#996699', '#66CCCC']

    for i, row in enumerate(data1):
        ax1.plot(row, label=header[i], linewidth=0.3, color=colors[i])
    for i, value in enumerate(content1):
        if value == 1:
            ax1.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_title('length d = 50, overlap a = 30%')

    for i, row in enumerate(data2):
        ax2.plot(row, label=header[i], linewidth=0.3, color=colors[i])
    for i, value in enumerate(content2):
        if value == 1:
            ax2.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_title('length d = 50, overlap a = 70%')

    for i, row in enumerate(data3):
        ax3.plot(row, label=header[i], linewidth=0.3, color=colors[i])
    for i, value in enumerate(content3):
        if value == 1:
            ax3.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.set_title('length d = 200, overlap a = 30%')

    for i, row in enumerate(data4):
        ax4.plot(row, label=header[i], linewidth=0.3, color=colors[i])
    for i, value in enumerate(content4):
        if value == 1:
            ax4.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax4.set_title('length d = 200, overlap a = 70%')

    # plt.legend(bbox_to_anchor=(0.1, 0.1), loc='upper left')
    fig.set_size_inches(20, 8)
    fig.suptitle(f'{filename}' + '_detected', fontsize=16, fontweight='bold')
    plt.subplots_adjust(hspace=0.8)
    plt.legend(loc=2, bbox_to_anchor=(1.005, 1.6))

    plt.show()
    # plt.savefig('../data/biobankdata_result/' + f'{filename}.png')
    # plt.close()






if __name__ == '__main__':

    draw_4_plots('sin_with_noise_events')
