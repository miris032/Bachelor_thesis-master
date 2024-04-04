from src import data_filling, binning, run, run_competitors, draw_slidSHAPs_plot_stock, draw_stock_timeseries
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


def main(file, bins, d, a, HDDDM_window_length):

    if any(file.startswith(prefix) for prefix in ('Open', 'Close', 'High', 'Low', 'Volume', 'Adj Close')):
        single_stock = False
    else:
        single_stock = True

    # read .csv file to the numpy array and binning operation
    if single_stock:
        time_category = file.split('_')[1]
        file_name = file.split('_')[0]
        data = np.array(pd.read_csv(f'{root}/data/ticker_data/{time_category}/{file_name}.csv', delimiter=",", header=None))
        data_binned = binning.binning(data, bins)
    else:
        data = np.array(pd.read_csv(f'{root}/data/ticker_data/{file}.csv', delimiter=",", header=None))
        data_binned = binning.binning(data, bins)
    np.savetxt(f'../data/Realworld_{file}_binned.csv', data_binned, delimiter=",", fmt='%s')

    # run slidSHAPs
    print(file)
    run.main2(f'{file}_binned')

    # plot slidSHAPs
    draw_slidSHAPs_plot_stock.draw_4_plots(f'{file}_binned', (d[0], d[1]), (a[0], a[1]))

    # run competitor
    data = data[1:, 1:].astype(float)
    data = pd.DataFrame(data)

    HDDDM_result = run_competitors.run_HDDDM_competitor(int(HDDDM_window_length), data)
    # KdqTree_result = run_competitors.run_KdqTree_competitor(int(KdqTree_window_length), data)
    ADWIN_result = run_competitors.run_ADWIN_competitor(0, data)
    # np.savetxt(f'../results/exp_2023_ijcai/{file}_binned/full-1/KdqTree_bi_predictions_{file}.csv', KdqTree_result, delimiter=",", fmt='%d')
    np.savetxt(f'../results/exp_2023_ijcai/{file}_binned/full-1/HDDDM_bi_predictions_{file}.csv', HDDDM_result, delimiter=",", fmt='%d')
    np.savetxt(f'../results/exp_2023_ijcai/{file}_binned/full-1/ADWIN_bi_predictions_{file}.csv', ADWIN_result, delimiter=",", fmt='%d')
    print()

    # plot time series with relocated slidSHAPs and competitors results
    draw_stock_timeseries.plot_timeseries(file, d[0], int(a[0]*d[0]/100))


if __name__ == '__main__':

    main('BMW.DE_daily', 100, (30, 50), (30, 70), 30)
    # main('BMW.DE_hourly', 100, (30, 50), (30, 70), 30)
    # main('BMW.DE_minutely', 100, (10, 15), (30, 70), 15)
    # main('Open_hourly', 100, (30, 50), (30, 70), 30)

    # main('BMW.DE(events)_daily', 100, (30, 50), (30, 70), 30)
    # main('BMW.DE(events)_hourly', 100, (30, 50), (30, 70), 30)
    # main('BMW.DE(events)_minutely', 100, (10, 15), (30, 70), 15)
    # main('Open_hourly(events)', 100, (30, 50), (30, 70), 30)

    # main('BMW.DE(events2)_daily', 100, (30, 50), (30, 70), 30)
    # main('BMW.DE(events2)_hourly', 100, (30, 50), (30, 70), 30)
    # main('BMW.DE(events2)_minutely', 100, (10, 15), (30, 70), 15)
    # main('Open_hourly(events2)', 100, (30, 50), (30, 70), 30)



