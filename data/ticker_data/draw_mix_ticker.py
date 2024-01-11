import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime



root = Path(__file__).resolve().parent.parent




def draw_plots(file):
    filename = os.path.basename(file)

    data = np.asarray(pd.read_csv(f'{root}/ticker_data/' + f'{file}' + '.csv', delimiter=",", header=None))
    data = np.transpose(data)
    time = data[0, 1:]
    # data = data[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 1:]
    data = data[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1:]
    data = data.astype(float)
    input_format = "%Y-%m-%d %H:%M:%S%z"
    output_format = "%Y-%m-%d %H:%M:%S%z"
    time = [datetime.strptime(date, input_format).strftime(output_format) for date in time]
    x_time2 = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S%z') for d in time]

    plt.figure(figsize=(20, 5))
    header = (f'1COV.DE_{filename}', f'ADS.DE_{filename}', f'AIR.DE_{filename}', f'ALV.DE_{filename}', f'BAS.DE_{filename}', f'BMW.DE_{filename}', f'CBK.DE_{filename}', f'DTE.DE_{filename}', f'EOAN.DE_{filename}', f'HEI.DE_{filename}',
              f'HNR1.DE_{filename}', f'IFX.DE_{filename}', f'MRK.DE_{filename}', f'RHM.DE_{filename}', f'RWE.DE_{filename}', f'SAP.DE_{filename}', f'SIE.DE_{filename}', f'SY1.DE_{filename}', f'VNA.DE_{filename}')
    colors = ['#FF9933', '#FF6666', '#66FF66', '#6666FF', '#996699', '#66CCCC']

    for i, row in enumerate(data):
        # plt.plot(x_time2, row, label=header[i], linewidth=0.3)
        plt.plot(row, label=header[i], linewidth=0.3)

    plt.suptitle(f'{filename}' + '', fontsize=16, fontweight='bold')
    plt.subplots_adjust(hspace=0.8)
    plt.legend(loc=2, bbox_to_anchor=(1, 1), prop={'size': 6})
    # plt.legend(loc=4)

    plt.show()
    # plt.savefig('../data/ticker_data_result/' + f'{filename}.png')
    # plt.close()


if __name__ == '__main__':
    draw_plots('High_hourly_events')
