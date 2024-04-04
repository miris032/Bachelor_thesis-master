from src import data_filling, binning, run, run_competitors, draw_slidSHAPs_plot_biobank, draw_biobank_timeseries
from pathlib import Path
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
root = Path(__file__).resolve().parent.parent
import json
import jsbeautifier
import sys


def main(file, bins, HDDDM_window_length):

    if file.startswith('dataset'):
        dataset_1_to_9 = True
    elif file.startswith('accs'):
        dataset_1_to_9 = False
    else:
        sys.exit('Unexpected data')

    # read .csv file to the pandas dataframe
    df = pd.read_csv(f'{root}/data/biobankdata/{file}.csv', delimiter=",", header=None)

    # fill the missing values
    if dataset_1_to_9:
        data = data_filling.single_dataset_filling(df)
    else:
        data = data_filling.accs_filling(df)
    np.savetxt(f'../data/biobankdata/{file}_filled.csv', data, delimiter=", ", fmt='%s')

    # binning operation
    data_binned = binning.binning(data, bins)
    np.savetxt(f'../data/Realworld_{file}_filled_binned.csv', data_binned, delimiter=",", fmt='%s')

    # run slidSHAPs
    run.main2(f'{file}_filled_binned')

    # plot slidSHAPs
    draw_slidSHAPs_plot_biobank.draw_4_plots(f'{file}_filled_binned', (100, 200), (30, 70))

    # run competitor
    data = data[1:, 1:].astype(float)
    data = pd.DataFrame(data)

    HDDDM_result = run_competitors.run_HDDDM_competitor(int(HDDDM_window_length), data)
    # KdqTree_result = run_competitors.run_KdqTree_competitor(int(KdqTree_window_length), data)
    ADWIN_result = run_competitors.run_ADWIN_competitor(0, data)
    # np.savetxt(f'../results/exp_2023_ijcai/{file}_filled_binned/full-1/KdqTree_bi_predictions_{file}_filled.csv', KdqTree_result, delimiter=",", fmt='%d')
    np.savetxt(f'../results/exp_2023_ijcai/{file}_filled_binned/full-1/HDDDM_bi_predictions_{file}_filled.csv', HDDDM_result, delimiter=",", fmt='%d')
    np.savetxt(f'../results/exp_2023_ijcai/{file}_filled_binned/full-1/ADWIN_bi_predictions_{file}_filled.csv', ADWIN_result, delimiter=",", fmt='%d')
    print()

    # plot time series with relocated slidSHAPs and competitors results
    draw_biobank_timeseries.plot_timeseries(file, 100, 70)


if __name__ == '__main__':

    main('dataset2-timeSeries', 50, 100)
    # main('accs', 50, 100)

    # main('dataset1-timeSeries(events2)', 50, 100)
    # main('dataset1-timeSeries(events)', 50, 100)
    # main('accs(event)', 50, 100)
    # main('accs(event2)', 50, 100)

