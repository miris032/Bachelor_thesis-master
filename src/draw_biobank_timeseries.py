import csv
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from textwrap import wrap
from src.util2 import getPosition

root = Path(__file__).resolve().parent.parent
'''insert_starts = (2501, 6751, 11001, 15251, 19501)
insert_ends = (3251, 7501, 11751, 16001, 20251)'''

#  = (3000, 6000, 9000, 12000, 15000)
# insert_ends = (3750, 6750, 9750, 12750, 15750)

insert_starts = (3000, 3700,   6000, 6700,   9000, 9700,   12000, 12700,   15000, 15700)
insert_ends = (3100, 3800,   6100, 6800,   9100, 9800,   12100, 12800,   15100, 15800)

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
    KdqTree = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_filled_binned/full-1/KdqTree_bi_predictions_{file}_filled.csv', delimiter=",", header=None).iloc[:, 0].tolist()
    HDDDM = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_filled_binned/full-1/HDDDM_bi_predictions_{file}_filled.csv', delimiter=",", header=None).iloc[:, 0].tolist()
    ADWIN = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_filled_binned/full-1/ADWIN_bi_predictions_{file}_filled.csv', delimiter=",", header=None).iloc[:, 0].tolist()

    if dataset_1_to_9:
        header = ['acc', 'light', 'moderate-vigorous', 'sedentary', 'sleep', 'MET']
        colors = ['lightblue', '#9784B6', '#D08AB9', '#FF94A8', '#BD7C6F', '#FFCE71']
    else:
        header = ('acc1', 'acc2', 'acc3', 'acc4', 'acc5', 'acc6', 'acc7', 'acc8', 'acc9')

    plt.figure(figsize=(20, 4))
    for i, row in enumerate(data1):
        plt.plot(row, color=colors[i], label=header[i], linewidth=0.5) if dataset_1_to_9 else plt.plot(row, label=header[i], linewidth=0.3)
    '''for i, value in enumerate(ts_predictions):
        if value == 1:
            plt.axvline(x=i, color='red', linestyle='--', linewidth=1)'''
    '''for i, value in enumerate(HDDDM):
        if value == 1:
            plt.axvline(x=i, color='blue', linestyle='--', linewidth=0.8, alpha=1)
    for i, value in enumerate(KdqTree):
            if value == 1:
                plt.axvline(x=i, color='orange', linestyle='--', linewidth=0.8, alpha=1)'''
    '''for i, value in enumerate(ADWIN):
        if value == 1:
            plt.axvline(x=i, color='green', linestyle='--', linewidth=0.8, alpha=1)'''

    # shadow areas
    ylim = plt.gca().get_ylim()
    if events:
        for j in range(len(insert_starts)):
            plt.fill_between(np.linspace(insert_starts[j], insert_ends[j]), 0, ylim[1], color='gray', hatch='///', alpha=0.2)

        # add the handle for the red dash line
        existing_handles, existing_labels = plt.gca().get_legend_handles_labels()
        new_handles = [plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='slidSHAPs detection'),
                       plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=1, label='HDDDM detection'),
                       plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=1, label='KdqTree detection'),
                       plt.Line2D([0], [0], color='green', linestyle='--', linewidth=1, label='ADWIN detection')]
        all_handles = existing_handles + new_handles
        all_labels = existing_labels + ['slidSHAPs detection', 'HDDDM detection', 'KdqTree detection', 'ADWIN detection']
        labels = ['\n'.join(wrap(label, 10)) for label in all_labels]
        plt.legend(handles=all_handles, labels=labels, loc='upper left', bbox_to_anchor=(1.005, 1.075), prop={'size': 16})
    else:
        existing_handles, existing_labels = plt.gca().get_legend_handles_labels()
        new_handles = [plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='slidSHAPs detection')]
        all_handles = existing_handles + new_handles
        all_labels = existing_labels + ['slidSHAPs detection']
        labels = ['\n'.join(wrap(label, 10)) for label in all_labels]
        plt.legend(handles=all_handles, labels=labels, loc='upper left', bbox_to_anchor=(1.005, 1.5), prop={'size': 16})

    plt.subplots_adjust(top=0.7, bottom=0.3, left=0.025, right=0.875, hspace=0.76)

    plt.savefig(f'../data/biobankdata_plots/{file}_filled_relocation.png')
    # plt.show()
    print(f'time series is saved in data/biobankdata_plots/{file}_filled_relocation.png')
    plt.close()


if __name__ == '__main__':

    # plot_timeseries('dataset1-timeSeries', 100, 70)
    # plot_timeseries('accs', 100, 70)

    # plot_timeseries('dataset1-timeSeries(events2)', 100, 70)
    plot_timeseries('accs(events2)', 100, 70)

