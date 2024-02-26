import csv
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.util2 import getPosition

root = Path(__file__).resolve().parent.parent
insert_starts = (2501, 6751, 11001, 15251, 19501)
insert_ends = (3251, 7501, 11751, 16001, 20251)


def plot_timeseries(file, d, a):

    events = '(events)' in file or '(events2)' in file
    dataset_1_to_9 = file.startswith('dataset')

    data = np.asarray(pd.read_csv(f'{root}/data/biobankdata/{file}_filled.csv', delimiter=",", header=None))
    data = data[1:]
    data = np.array(data)
    data = np.transpose(data)
    data1 = data[1:, :]
    data1 = data1.astype(float)

    pkl = pd.read_pickle(f'{root}/results/exp_2023_ijcai/{file}_filled_binned/full-1/ts_drifts_bi_predictions_Realworld_{file}_filled_binned_windowlength{d}_overlap{a}.pkl')
    ts_predictions = list(pkl.values())[0]
    print(getPosition(ts_predictions))
    #DDM = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_filled_binned/full-1/DDM_bi_predictions_{file}_filled_binned.csv', delimiter=",", header=None).iloc[:, 0].tolist()
    HDDDM = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_filled_binned/full-1/HDDDM_bi_predictions_{file}_filled_binned.csv', delimiter=",", header=None).iloc[:, 0].tolist()
    ADWIN = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_filled_binned/full-1/ADWIN_bi_predictions_{file}_filled_binned.csv', delimiter=",", header=None).iloc[:, 0].tolist()

    if dataset_1_to_9:
        header = ['acc', 'light', 'MVPA', 'sedentary', 'sleep', 'MET']
        colors = ['lightblue', '#9784B6', '#D08AB9', '#FF94A8', '#BD7C6F', '#FFCE71']
    else:
        header = ('acc1', 'acc2', 'acc3', 'acc4', 'acc5', 'acc6', 'acc7', 'acc8', 'acc9')

    plt.figure(figsize=(20, 2))
    for i, row in enumerate(data1):
        plt.plot(row, color=colors[i], label=header[i], linewidth=0.5) if dataset_1_to_9 else plt.plot(row, label=header[i], linewidth=0.3)
    for i, value in enumerate(ts_predictions):
        if value == 1:
            plt.axvline(x=i, color='red', linestyle='--', linewidth=1)
    '''for i, value in enumerate(DDM):
        if value == 1:
            plt.axvline(x=i, color='orange', linestyle='--', linewidth=0.8, alpha=1)'''
    for i, value in enumerate(HDDDM):
        if value == 1:
            plt.axvline(x=i, color='blue', linestyle='--', linewidth=0.8, alpha=1)
    for i, value in enumerate(ADWIN):
        if value == 1:
            plt.axvline(x=i, color='green', linestyle='--', linewidth=0.8, alpha=1)

    # shadow areas
    ylim = plt.gca().get_ylim()
    if events:
        for j in range(len(insert_starts)):
            plt.fill_between(np.linspace(insert_starts[j], insert_ends[j]), 0, ylim[1], color='gray', hatch='///', alpha=0.2)

        # add the handle for the red dash line
        existing_handles, existing_labels = plt.gca().get_legend_handles_labels()
        new_handles = [plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='slidSHAPs detection'),
                       plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=1, label='DDM detection'),
                       plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=1, label='HDDDM detection'),
                       plt.Line2D([0], [0], color='green', linestyle='--', linewidth=1, label='ADWIN detection')]
        all_handles = existing_handles + new_handles
        all_labels = existing_labels + ['slidSHAPs detection', 'DDM detection', 'HDDDM detection', 'ADWIN detection']
        plt.legend(handles=all_handles, labels=all_labels, loc='upper left', bbox_to_anchor=(1.005, 1.1),
                   prop={'size': 8})
        plt.subplots_adjust(hspace=0.8)

    plt.savefig(f'../data/biobankdata_plots/{file}_filled_relocation.png')
    # plt.show()
    print(f'time series is saved in data/biobankdata_plots/{file}_filled_relocation.png')
    plt.close()


if __name__ == '__main__':

    plot_timeseries('dataset1-timeSeries', 100, 70)
    # plot_timeseries('dataset1-timeSeries(events2)', 100, 70)
    # plot_timeseries('accs(events2)', 100, 70)

