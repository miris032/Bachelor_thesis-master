import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.ddm import DDM
from menelaus.data_drift.hdddm import HDDDM
from pathlib import Path
from src import run_competitors
from src.run_competitors import run_HDDDM_competitor, run_ADWIN_competitor, run_DDM_competitor, run_KdqTree_competitor, run_PCACD_competitor
from src.util2 import getPosition

root = Path(__file__).resolve().parent.parent

if __name__ == '__main__':

    '''np.random.seed(42)
    data_stream_1 = np.random.normal(loc=5, scale=1, size=(500, 3))  # 修改均值以避免接近0
    data_stream_2 = np.random.normal(loc=10, scale=1, size=(500, 3))
    data_stream_3 = np.random.normal(loc=15, scale=1, size=(500, 3))
    # combine data_stream_1 and data_stream_2 to create the drift
    data_stream = np.vstack((data_stream_1, data_stream_2, data_stream_3))
    data_stream = pd.DataFrame(data_stream).astype(float)'''

    # data_stream = np.asarray(pd.read_csv(f'{root}/data/biobankdata/accs(events2)_filled.csv', delimiter=",", header=None))
    data_stream = np.asarray(pd.read_csv(f'{root}/data/ticker_data/daily/BMW.DE.csv', delimiter=",", header=None))
    # data_stream = np.asarray(pd.read_csv(f'{root}/data/ticker_data/Open_hourly.csv', delimiter=",", header=None))

    data_stream = pd.DataFrame(data_stream).iloc[1:, 1:].astype(float)
    # print(data_stream)


    # slidSHAPs
    # pkl = pd.read_pickle(f'{root}/results/exp_2023_ijcai/accs(events2)_filled_binned/full-1/ts_drifts_bi_predictions_Realworld_accs(events2)_filled_binned_windowlength100_overlap70.pkl')

    pkl = pd.read_pickle(f'{root}/results/exp_2023_ijcai/BMW.DE(events2)_daily_binned/full-1/ts_drifts_bi_predictions_Realworld_BMW.DE(events2)_daily_binned_windowlength50_overlap35.pkl')
    ts_predictions = list(pkl.values())[0]
    print(f'slidSHAPs: Change detected at index: {getPosition(ts_predictions)}')
    print(len(getPosition(ts_predictions)))
    print()

    # ADWIN
    ADWIN = run_ADWIN_competitor(100, data_stream)
    print(f'ADWIN: Change detected at index: {getPosition(ADWIN)}')
    print(f'ADWIN: Detected change number: {len(getPosition(ADWIN))}')
    print()

    # HDDDM
    HDDDM_window_size = 50
    HDDDM = run_HDDDM_competitor(HDDDM_window_size, data_stream)
    print(f'HDDDM: Change detected at index: {getPosition(HDDDM)}')
    print(len(getPosition(HDDDM)))
    print()

    # Kdq-tree
    kdqtree = run_KdqTree_competitor(200, data_stream)
    print(f'Kdq-tree: Change detected at index: {getPosition(kdqtree)}')
    print(len(getPosition(kdqtree)))

    '''# PCACD
    pcacd = run_PCACD_competitor(300, data_stream)
    print(f'PCACD: Change detected at index: {getPosition(pcacd)}')
    print(len(getPosition(pcacd)))'''






    # Visualisation of the data stream
    plt.figure(figsize=(10, 5))
    for column in data_stream.columns:
        plt.plot(data_stream[column], label=column, linewidth=0.3, alpha=0.5)
    plt.title('Generated Data Stream')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
