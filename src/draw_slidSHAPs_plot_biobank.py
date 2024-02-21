import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
root = Path(__file__).resolve().parent.parent


def ts_pos_to_slid_pos(pos_list, d, a):
    slid_pos = []
    for pos in pos_list:
        print(pos)
        slid_pos.append((pos-d) / (d-d*a/100))
        print(f'({pos}-{d}) / ({d}-{d}*{a}/100) = {(pos-d) / (d-d*a/100)}')
    print(slid_pos)
    return slid_pos


def draw_4_plots(file):

    events = '(events)' in file or '(events2)' in file
    dataset_1_to_9 = file.startswith('dataset')

    filename = os.path.basename(file)
    data1 = np.asarray(pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshaps_Realworld_{file}_windowlength50_overlap15.txt', delimiter=",", header=None))
    data2 = np.asarray(pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshaps_Realworld_{file}_windowlength50_overlap35.txt', delimiter=",", header=None))
    data3 = np.asarray(pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshaps_Realworld_{file}_windowlength200_overlap60.txt', delimiter=",", header=None))
    data4 = np.asarray(pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshaps_Realworld_{file}_windowlength200_overlap140.txt', delimiter=",", header=None))

    pkl1 = pd.read_pickle(f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_windowlength50_overlap15.pkl')
    pkl2 = pd.read_pickle(f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_windowlength50_overlap35.pkl')
    pkl3 = pd.read_pickle(f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_windowlength200_overlap60.pkl')
    pkl4 = pd.read_pickle(f'{root}/results/exp_2023_ijcai/{file}/full-1/slidshap_drifts_bi_predictions_Realworld_{file}_windowlength200_overlap140.pkl')
    slidshap_prediction1 = list(pkl1.values())[0]
    slidshap_prediction2 = list(pkl2.values())[0]
    slidshap_prediction3 = list(pkl3.values())[0]
    slidshap_prediction4 = list(pkl4.values())[0]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

    header = (
        ('acc_shap', 'light_shap', 'mv_shap', 'sedentary_shap', 'sleep_shap', 'MET_shap')
        if dataset_1_to_9
        else ('acc1', 'acc2', 'acc3', 'acc4', 'acc5', 'acc6', 'acc7', 'acc8', 'acc9')
    )
    colors = (
        ['#FF6666', '#6666FF', '#66FF66', '#FF9933', '#996699', '#66CCCC']
        if file.startswith('dataset')
        else ['#FF6666', '#6666FF', '#66FF66', '#FF9933', '#996699', '#66CCCC', '#FFCC66', '#CC66FF', '#99CC66']
    )

    for i, row in enumerate(data1):
        ax1.plot(row, label=header[i], linewidth=0.3, color=colors[i])
    for i, value in enumerate(slidshap_prediction1):
        if value == 1:
            ax1.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=1)
    ax1.set_title('length d = 50, overlap a = 30%')

    for i, row in enumerate(data2):
        ax2.plot(row, label=header[i], linewidth=0.3, color=colors[i])
    for i, value in enumerate(slidshap_prediction2):
        if value == 1:
            ax2.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=1)
    ax2.set_title('length d = 50, overlap a = 70%')

    for i, row in enumerate(data3):
        ax3.plot(row, label=header[i], linewidth=0.3, color=colors[i])
    for i, value in enumerate(slidshap_prediction3):
        if value == 1:
            ax3.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=1)
    ax3.set_title('length d = 200, overlap a = 30%')

    for i, row in enumerate(data4):
        ax4.plot(row, label=header[i], linewidth=0.3, color=colors[i])
    for i, value in enumerate(slidshap_prediction4):
        if value == 1:
            ax4.axvline(x=i, color='red', linestyle='--', linewidth=0.8, alpha=1)
    ax4.set_title('length d = 200, overlap a = 70%')

    # shadow areas
    if events:
        insert_starts = (2501, 7501, 12501, 17501, 22501)
        insert_ends = (4000, 9000, 14000, 19000, 24000)
        ylim = plt.gca().get_ylim()

        start_slidSHAPs = ts_pos_to_slid_pos(insert_starts, 50, 30)
        end_slidSHAPs = ts_pos_to_slid_pos(insert_ends, 50, 30)
        for j in range(len(insert_starts)):
            ax1.fill_between(np.linspace(start_slidSHAPs[j], end_slidSHAPs[j]), ylim[0], ylim[1], color='gray', hatch='///', alpha=0.2)

        start_slidSHAPs = ts_pos_to_slid_pos(insert_starts, 50, 70)
        end_slidSHAPs = ts_pos_to_slid_pos(insert_ends, 50, 70)
        for j in range(len(insert_starts)):
            ax2.fill_between(np.linspace(start_slidSHAPs[j], end_slidSHAPs[j]), ylim[0], ylim[1], color='gray', hatch='///', alpha=0.2)

        start_slidSHAPs = ts_pos_to_slid_pos(insert_starts, 200, 30)
        end_slidSHAPs = ts_pos_to_slid_pos(insert_ends, 200, 30)
        for j in range(len(insert_starts)):
            ax3.fill_between(np.linspace(start_slidSHAPs[j], end_slidSHAPs[j]), ylim[0], ylim[1], color='gray', hatch='///', alpha=0.2)

        start_slidSHAPs = ts_pos_to_slid_pos(insert_starts, 200, 70)
        end_slidSHAPs = ts_pos_to_slid_pos(insert_ends, 200, 70)
        for j in range(len(insert_starts)):
            ax4.fill_between(np.linspace(start_slidSHAPs[j], end_slidSHAPs[j]), ylim[0], ylim[1], color='gray', hatch='///', alpha=0.2)

    # plt.legend(bbox_to_anchor=(0.1, 0.1), loc='upper left')
    fig.set_size_inches(20, 8)
    fig.suptitle(f'{filename}_detection', fontsize=16, fontweight='bold')
    plt.subplots_adjust(hspace=0.8)

    # add the handle for the red dash line
    existing_handles, existing_labels = plt.gca().get_legend_handles_labels()
    new_handles = [plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='slidSHAPs detection')]
    all_handles = existing_handles + new_handles
    all_labels = existing_labels + ['slidSHAPs detection']

    plt.legend(handles=all_handles, labels=all_labels, loc='upper left', bbox_to_anchor=(1.005, 6.4), prop={'size': 8})
    plt.savefig(f'../data/biobankdata_plots/{filename}.png')
    # plt.show()
    print(f'slidSHAPs result is saved in data/biobankdata_plots/{filename}.png')
    print()
    plt.close()


if __name__ == '__main__':

    draw_4_plots('accs_after_filling_and_binning')







