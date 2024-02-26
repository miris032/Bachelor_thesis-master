import csv
import os
from pathlib import Path
from src.util2 import ts_pos_to_slid_pos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
root = Path(__file__).resolve().parent.parent


def draw_4_plots(file, l, a):

    events = '(events)' in file or '(events2)' in file
    dataset_1_to_9 = file.startswith('dataset')

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

    if dataset_1_to_9:
        header = ('acc_shap', 'light_shap', 'mv_shap', 'sedentary_shap', 'sleep_shap', 'MET_shap')
        colors = ('#FF6666', '#6666FF', '#66FF66', '#FF9933', '#996699', '#66CCCC')
    else:
        header = ('acc1', 'acc2', 'acc3', 'acc4', 'acc5', 'acc6', 'acc7', 'acc8', 'acc9')
        colors = ['#FF6666', '#6666FF', '#66FF66', '#FF9933', '#996699', '#66CCCC', '#FFCC66', '#CC66FF', '#99CC66']

    for i, row in enumerate(data1):
        ax1.plot(row, label=header[i], linewidth=0.5, color=colors[i])
    for i, value in enumerate(slidshap_prediction1):
        if value == 1:
            ax1.axvline(x=i, color='red', linestyle='--', linewidth=1)
    ax1.set_title(f'length d = {l[0]}, overlap a = {a[0]}%')

    for i, row in enumerate(data2):
        ax2.plot(row, label=header[i], linewidth=0.5, color=colors[i])
    for i, value in enumerate(slidshap_prediction2):
        if value == 1:
            ax2.axvline(x=i, color='red', linestyle='--', linewidth=1)
    ax2.set_title(f'length d = {l[0]}, overlap a = {a[1]}%')

    for i, row in enumerate(data3):
        ax3.plot(row, label=header[i], linewidth=0.5, color=colors[i])
    for i, value in enumerate(slidshap_prediction3):
        if value == 1:
            ax3.axvline(x=i, color='red', linestyle='--', linewidth=1)
    ax3.set_title(f'length d = {l[1]}, overlap a = {a[0]}%')

    for i, row in enumerate(data4):
        ax4.plot(row, label=header[i], linewidth=0.5, color=colors[i])
    for i, value in enumerate(slidshap_prediction4):
        if value == 1:
            ax4.axvline(x=i, color='red', linestyle='--', linewidth=1)
    ax4.set_title(f'length d = {l[1]}, overlap a = {a[1]}%')

    # shadow areas
    if events:
        insert_starts = (2501, 6751, 11001, 15251, 19501)
        insert_ends = (3251, 7501, 11751, 16001, 20251)
        ylim = plt.gca().get_ylim()

        for i, axi in enumerate([ax1, ax2]):
            start_slidSHAPs = ts_pos_to_slid_pos(insert_starts, l[0], a[i])
            end_slidSHAPs = ts_pos_to_slid_pos(insert_ends, l[0], a[i])
            for j in range(len(insert_starts)):
                print(f'{start_slidSHAPs[j]}-{end_slidSHAPs[j]}')
                axi.fill_between(np.linspace(start_slidSHAPs[j], end_slidSHAPs[j]), ylim[0], ylim[1], color='gray', hatch='///', alpha=0.2)
        print()
        for i, axi in enumerate([ax3, ax4]):
            start_slidSHAPs = ts_pos_to_slid_pos(insert_starts, l[1], a[i])
            end_slidSHAPs = ts_pos_to_slid_pos(insert_ends, l[1], a[i])
            for j in range(len(insert_starts)):
                print(f'{start_slidSHAPs[j]}-{end_slidSHAPs[j]}')
                axi.fill_between(np.linspace(start_slidSHAPs[j], end_slidSHAPs[j]), ylim[0], ylim[1], color='gray', hatch='///', alpha=0.2)


    # plt.legend(bbox_to_anchor=(0.1, 0.1), loc='upper left')
    fig.set_size_inches(20, 8)
    plt.subplots_adjust(hspace=0.8)

    # add the handle for the red dash line
    existing_handles, existing_labels = plt.gca().get_legend_handles_labels()
    new_handles = [plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='slidSHAPs detection')]
    all_handles = existing_handles + new_handles
    all_labels = existing_labels + ['slidSHAPs detection']
    plt.legend(handles=all_handles, labels=all_labels, loc='upper left', bbox_to_anchor=(1.005, 6.4), prop={'size': 8})

    plt.savefig(f'../data/biobankdata_plots/slidshap_{filename}.png')
    # plt.show()
    print(f'slidSHAPs result is saved in data/biobankdata_plots/slidshap_{filename}.png')
    print()
    plt.close()


if __name__ == '__main__':

    draw_4_plots('dataset1-timeSeries_filled', (100, 200), (30, 70))
    # draw_4_plots('accs_filled_binned')







