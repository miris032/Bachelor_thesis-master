import csv
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
insert_starts = (2501, 7501, 12501, 17501, 22501)
insert_ends = (4000, 9000, 14000, 19000, 24000)


# input: 0s, 1s list; output: position of 1s
def getPosition(input_list):
    drift_pos = []
    for i, values in enumerate(input_list):
        if values == 1:
            drift_pos.append(i)
    return drift_pos


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
    HDDDM = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_filled_binned/full-1/HDDDM_bi_predictions_{file}_filled_binned.csv', delimiter=",", header=None).iloc[:, 0].tolist()

    if dataset_1_to_9:
        header = ['acc', 'light', 'MVPA', 'sedentary', 'sleep', 'MET']
        colors = ['lightblue', '#FFCCCC', '#FFDDBB', '#E6E6FA', '#FF94A8', '#FD94A8']
    else:
        header = ('acc1', 'acc2', 'acc3', 'acc4', 'acc5', 'acc6', 'acc7', 'acc8', 'acc9')

    plt.figure(figsize=(20, 2))
    for i, row in enumerate(data1):
        plt.plot(row, color=colors[i], label=header[i], linewidth=0.3) if dataset_1_to_9 else plt.plot(row, label=header[i], linewidth=0.3)
    for i, value in enumerate(ts_predictions):
        if value == 1:
            plt.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=1)

    # shadow areas
    if events:
        for j in range(len(insert_starts)):
            plt.fill_between(np.linspace(insert_starts[j], insert_ends[j]), 0, 400, color='gray', hatch='///', alpha=0.2)

    # add the handle for the red dash line
    existing_handles, existing_labels = plt.gca().get_legend_handles_labels()
    new_handles = [plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='slidSHAPs detection')]
    all_handles = existing_handles + new_handles
    all_labels = existing_labels + ['slidSHAPs detection']

    plt.legend(handles=all_handles, labels=all_labels, loc='upper left', bbox_to_anchor=(1.005, 1), prop={'size': 8})
    plt.savefig(f'../data/biobankdata_plots/{file}_filled.png')
    # plt.show()
    print(f'time series is saved in data/biobankdata_plots/{file}_filled.png')
    plt.close()


if __name__ == '__main__':
    # plot_timeseries('dataset1-timeSeries(events2)', 200, 140)
    plot_timeseries('accs(events2)', 200, 140)

