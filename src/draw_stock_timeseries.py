import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
root = Path(__file__).resolve().parent.parent


def plot_timeseries(file, d, a):

    events = '(events)' in file or '(events2)' in file
    mix = any(file.startswith(prefix) for prefix in ('Open', 'Close', 'High', 'Low', 'Volume', 'Adj Close'))

    if not mix:
        time_category = file.split('_')[1]
        file_name = file.split('_')[0]
        data = np.asarray(pd.read_csv(f'{root}/data/ticker_data/{time_category}/{file_name}.csv', delimiter=",", header=None))
        data = np.transpose(data)
        # time1 = data[0, 1:]
        data1 = data[[1, 2, 3, 4], 1:]
    else:
        data = np.asarray(pd.read_csv(f'{root}/data/ticker_data/{file}.csv', delimiter=",", header=None))
        data = np.transpose(data)
        # time = data[0, 1:]
        data1 = data[1:, 1:]
    data1 = data1.astype(float)

    pkl = pd.read_pickle(f'{root}/results/exp_2023_ijcai/{file}_binned/full-1/ts_drifts_bi_predictions_Realworld_{file}_binned_windowlength{d}_overlap{a}.pkl')
    ts_predictions = list(pkl.values())[0]
    #DDM = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_binned/full-1/DDM_bi_predictions_{file}_binned.csv', delimiter=",", header=None).iloc[:, 0].tolist()
    HDDDM = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_binned/full-1/HDDDM_bi_predictions_{file}_binned.csv', delimiter=",", header=None).iloc[:, 0].tolist()
    ADWIN = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_binned/full-1/ADWIN_bi_predictions_{file}_binned.csv', delimiter=",", header=None).iloc[:, 0].tolist()

    if not mix:
        header = ('Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume')
    else:
        header = (f'1COV.DE', f'ADS.DE', f'AIR.DE', f'ALV.DE', f'BAS.DE', f'BMW.DE', f'CBK.DE', f'DTE.DE', f'EOAN.DE', f'HEI.DE')

    plt.figure(figsize=(20, 2))
    for i, row in enumerate(data1):
        plt.plot(row, label=header[i], linewidth=0.5)
    for i, value in enumerate(ts_predictions):
        if value == 1:
            plt.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=1)
    '''for i, value in enumerate(DDM):
        if value == 1:
            plt.axvline(x=i, color='orange', linestyle='--', linewidth=0.8, alpha=1)'''
    for i, value in enumerate(HDDDM):
        if value == 1:
            plt.axvline(x=i, color='blue', linestyle='--', linewidth=0.8, alpha=1)
    for i, value in enumerate(ADWIN):
        if value == 1:
            plt.axvline(x=i, color='green', linestyle='--', linewidth=0.8, alpha=1)

    #ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # shadow areas
    ylim = plt.gca().get_ylim()
    if events:
        if time_category == 'daily':
            '''insert_starts = (1201, 2651, 4101, 5551, 7001)
            insert_ends = (1450, 2900, 4350, 5800, 7250)'''
            insert_starts = (1200, 2400, 3600, 4800, 6000)
            insert_ends = (1450, 2650, 3850, 5050, 6250)
        elif time_category == 'hourly':
            '''insert_starts = (1001, 2201, 3401, 4601, 5801)
            insert_ends = (1200, 2400, 3600, 4800, 6000)'''
            insert_starts = (1000, 2000, 3000, 4000, 5000)
            insert_ends = (1200, 2200, 3200, 4200, 5200)
        else:
            '''insert_starts = (301, 671, 1041, 1411, 1781)
            insert_ends = (370, 740, 1110, 1480, 1850)'''
            insert_starts = (300, 600, 900, 1200, 1500)
            insert_ends = (370, 670, 970, 1270, 1570)
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
        plt.legend(handles=all_handles, labels=all_labels, loc='upper left', bbox_to_anchor=(1.005, 1.4),
                   prop={'size': 8})
        plt.subplots_adjust(hspace=0.8)

    plt.savefig(f'../data/ticker_data_plots/{file}_relocation.png')
    # plt.show()
    print(f'time series is saved in data/biobankdata_plots/{file}_relocation.png')
    plt.close()


'''insert_starts = (1001, 2501, 4001, 5501, 7001)
insert_ends = (1500, 3000, 4500, 6000, 7500)'''




if __name__ == '__main__':

    plot_timeseries('Open_hourly_test', 50, 35)
    # plot_timeseries('BMW.DE(events)_minutely', 50, 35)
    # HDDDM = pd.read_csv(f'{root}/results/exp_2023_ijcai/BMW.DE(events)_minutely_binned/full-1/HDDDM_bi_predictions_BMW.DE(events)_minutely_binned.csv', delimiter=",", header=None).iloc[:, 0].tolist()

