import pickle
import numpy as np
from src.drift_detection import _slidSHAPPos_to_tsPos

if __name__ == '__main__':

    arr1 = []
    for i in range(1, 10):
        ol = int(50 * 0.1 * i)
        data = open(
            f'./slidshap_drifts_bi_predictions_Realworld_dataset1-timeSeries_after_filling_and_binning_windowlength50_overlap{ol}.pkl',
            'rb')
        content = list(pickle.load(data).values())[0]
        in_sum = np.sum(content)
        arr1.append(in_sum)
    print(f'd = 50: {arr1}')

    arr2 = []
    for i in range(1, 10):
        ol = int(100 * 0.1 * i)
        data = open(
            f'./slidshap_drifts_bi_predictions_Realworld_dataset1-timeSeries_after_filling_and_binning_windowlength100_overlap{ol}.pkl',
            'rb')
        content = list(pickle.load(data).values())[0]
        in_sum = np.sum(content)
        arr2.append(in_sum)
    print(f'd = 100: {arr2}')

    arr3 = []
    for i in range(1, 10):
        ol = int(200 * 0.1 * i)
        data = open(
            f'./slidshap_drifts_bi_predictions_Realworld_dataset1-timeSeries_after_filling_and_binning_windowlength200_overlap{ol}.pkl',
            'rb')
        content = list(pickle.load(data).values())[0]
        in_sum = np.sum(content)
        arr3.append(in_sum)
    print(f'd = 200: {arr3}')


