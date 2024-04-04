import csv
import math
import os
from pathlib import Path
from src.util2 import ts_poslist_to_slid_poslist, ts_pos_to_slid_pos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from textwrap import wrap
root = Path(__file__).resolve().parent.parent


def draw_4_plots(file, l, a):

    events = '(events)' in file or '(events2)' in file
    mix = any(file.startswith(prefix) for prefix in ('Open', 'Close', 'High', 'Low', 'Volume', 'Adj Close'))

    filename = os.path.basename(file)
    data1 = np.asarray(pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshaps_Realworld_{file}_windowlength{l[0]}_overlap{int(l[0] * a[0] /100)}.txt', delimiter=",", header=None))
    data2 = np.asarray(pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshaps_Realworld_{file}_windowlength{l[0]}_overlap{int(l[0] * a[1] /100)}.txt', delimiter=",", header=None))
    data3 = np.asarray(pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshaps_Realworld_{file}_windowlength{l[1]}_overlap{int(l[1] * a[0] /100)}.txt', delimiter=",", header=None))
    data4 = np.asarray(pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshaps_Realworld_{file}_windowlength{l[1]}_overlap{int(l[1] * a[1] /100)}.txt', delimiter=",", header=None))

    pkl1 = pd.read_pickle(f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_windowlength{l[0]}_overlap{int(l[0] * a[0] /100)}.pkl')
    pkl2 = pd.read_pickle(f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_windowlength{l[0]}_overlap{int(l[0] * a[1] /100)}.pkl')
    pkl3 = pd.read_pickle(f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_windowlength{l[1]}_overlap{int(l[1] * a[0] /100)}.pkl')
    pkl4 = pd.read_pickle(f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_windowlength{l[1]}_overlap{int(l[1] * a[1] /100)}.pkl')
    slidshap_prediction1 = list(pkl1.values())[0]
    slidshap_prediction2 = list(pkl2.values())[0]
    slidshap_prediction3 = list(pkl3.values())[0]
    slidshap_prediction4 = list(pkl4.values())[0]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

    if mix:
        header = (f'1COV.DE', f'ADS.DE', f'AIR.DE', f'ALV.DE', f'BAS.DE', f'BMW.DE', f'CBK.DE', f'DTE.DE', f'EOAN.DE', f'HEI.DE')
    elif file.split('_')[1] == 'daily':
        header = ('Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume')
    else:
        header = ('Open', 'High', 'Low', 'Close', 'Volume')
    colors = ['#FF6666', '#6666FF', '#66FF66', '#FF9933', '#996699', '#66CCCC', '#FFCC66', '#CC66FF', '#99CC66', '#D08AB9']

    for i, row in enumerate(data1):
        ax1.plot(row, label=header[i], linewidth=0.5, color=colors[i])
    for i, value in enumerate(slidshap_prediction1):
        if value == 1:
            ax1.axvline(x=i, color='red', linestyle='--', linewidth=1)
    ax1.set_title(f'length d = {l[0]}, overlap a = {a[0]}%', fontsize=18)

    for i, row in enumerate(data2):
        ax2.plot(row, label=header[i], linewidth=0.5, color=colors[i])
    for i, value in enumerate(slidshap_prediction2):
        if value == 1:
            ax2.axvline(x=i, color='red', linestyle='--', linewidth=1)
    ax2.set_title(f'length d = {l[0]}, overlap a = {a[1]}%', fontsize=18)

    for i, row in enumerate(data3):
        ax3.plot(row, label=header[i], linewidth=0.5, color=colors[i])
    for i, value in enumerate(slidshap_prediction3):
        if value == 1:
            ax3.axvline(x=i, color='red', linestyle='--', linewidth=1)
    ax3.set_title(f'length d = {l[1]}, overlap a = {a[0]}%', fontsize=18)

    for i, row in enumerate(data4):
        ax4.plot(row, label=header[i], linewidth=0.5, color=colors[i])
    for i, value in enumerate(slidshap_prediction4):
        if value == 1:
            ax4.axvline(x=i, color='red', linestyle='--', linewidth=1)
    ax4.set_title(f'length d = {l[1]}, overlap a = {a[1]}%', fontsize=18)

    # draw the events areas
    if '(events)' in file or '(events2)' in file:
        if file.split('_')[1] == 'daily':
            '''insert_starts = (1201, 2651, 4101, 5551, 7001)
            insert_ends = (1450, 2900, 4350, 5800, 7250)'''
            insert_starts = (1200, 2400, 3600, 4800, 6000)
            insert_ends = (1450, 2650, 3850, 5050, 6250)
            # insert_starts = (1200, 1450,   2400, 2650,   3600, 3850,   4800, 5050,   6000, 6250)
            # insert_ends = (1230, 1480,   2430, 2680,   3630, 3880,   4830, 5080,   6030, 6280)
        elif file.split('_')[1] == 'hourly':
            '''insert_starts = (1001, 2201, 3401, 4601, 5801)
            insert_ends = (1200, 2400, 3600, 4800, 6000)'''
            # insert_starts = (1000, 2000, 3000, 4000, 5000)
            # insert_ends = (1200, 2200, 3200, 4200, 5200)
            insert_starts = (1000, 1200,   2000, 2200,   3000, 3200,   4000, 4200,   5000, 5200)
            insert_ends = (1030, 1230,   2030, 2230,   3030, 3230,   4030, 4230,   5030, 5230)
        elif file.split('_')[1] == 'minutely':
            '''insert_starts = (301, 671, 1041, 1411, 1781)
            insert_ends = (370, 740, 1110, 1480, 1850)'''
            insert_starts = (300, 600, 900, 1200, 1500)
            insert_ends = (370, 670, 970, 1270, 1570)

        # mix
        else:
            '''insert_starts = (1001, 2201, 3401, 4601, 5801)
            insert_ends = (1200, 2400, 3600, 4800, 6000)'''
            # insert_starts = (1000, 2000, 3000, 4000, 5000)
            # insert_ends = (1200, 2200, 3200, 4200, 5200)
            insert_starts = (1000, 1200,   2000, 2200,   3000, 3200,   4000, 4200,   5000, 5200)
            insert_ends = (1030, 1230,   2030, 2230,   3030, 3230,   4030, 4230,   5030, 5230)

        # shadow areas
        ylim = plt.gca().get_ylim()
        for i, axi in enumerate([ax1, ax2]):
            buffer_size = 5 * l[0]
            start_slidSHAPs = ts_poslist_to_slid_poslist(insert_starts, l[0], a[i])
            end_slidSHAPs = ts_poslist_to_slid_poslist(insert_ends, l[0], a[i])
            for j in range(len(insert_starts)):
                print(f'{start_slidSHAPs[j]}-{end_slidSHAPs[j]}')
                axi.fill_between(np.linspace(start_slidSHAPs[j], end_slidSHAPs[j]), 0, ylim[1], color='gray', hatch='///', alpha=0.2)
                axi.fill_between(np.linspace(end_slidSHAPs[j], end_slidSHAPs[j] + ts_pos_to_slid_pos(buffer_size, l[0], a[i])), 0, ylim[1], color='purple', hatch='///', alpha=0.1)
        print()
        for i, axi in enumerate([ax3, ax4]):
            buffer_size = 5 * l[1]
            start_slidSHAPs = ts_poslist_to_slid_poslist(insert_starts, l[1], a[i])
            end_slidSHAPs = ts_poslist_to_slid_poslist(insert_ends, l[1], a[i])
            for j in range(len(insert_starts)):
                print(f'{start_slidSHAPs[j]}-{end_slidSHAPs[j]}')
                axi.fill_between(np.linspace(start_slidSHAPs[j], end_slidSHAPs[j]), 0, ylim[1], color='gray', hatch='///', alpha=0.2)
                axi.fill_between(np.linspace(end_slidSHAPs[j], end_slidSHAPs[j] + ts_pos_to_slid_pos(buffer_size, l[1], a[i])), 0, ylim[1], color='purple', hatch='///', alpha=0.1)


    # plt.legend(bbox_to_anchor=(0.1, 0.1), loc='upper left')
    fig.set_size_inches(20, 8)
    plt.subplots_adjust(left=0.025, right=0.875, hspace=0.76)

    # add the handle for the red dash line
    existing_handles, existing_labels = plt.gca().get_legend_handles_labels()
    new_handles = [plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='slidSHAPs detection')]
    all_handles = existing_handles + new_handles
    all_labels = existing_labels + ['slidSHAPs detection']
    labels = ['\n'.join(wrap(label, 10)) for label in all_labels]
    plt.legend(handles=all_handles, labels=labels, loc='upper left', bbox_to_anchor=(1.005, 6.4), prop={'size': 16})

    plt.savefig(f'../data/ticker_data_plots/slidshap_{filename}.png')
    # plt.show()
    print(f'slidSHAPs result is saved in data/ticker_data_plots/slidshap_{filename}.png')
    print()
    plt.close()


if __name__ == '__main__':

    # draw_4_plots('BMW.DE_daily_binned', (30, 50), (30, 70))
    # draw_4_plots('Open_hourly_binned', (30, 50), (30, 70))

    draw_4_plots('BMW.DE(events2)_hourly_binned', (30, 50), (30, 70))
    # draw_4_plots('Open_hourly(events)_binned', (30, 50), (30, 70))



