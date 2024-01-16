import pickle
import numpy as np
from src.drift_detection import _slidSHAPPos_to_tsPos

if __name__ == '__main__':

    for i in range(1, 10):
        print(f'd = 50, a = {i * 10}%')
        ol = int(50 * 0.1 * i)
        data = open(
            f'./slidshap_drifts_bi_predictions_Realworld_dataset1-timeSeries_after_filling_and_binning_windowlength50_overlap{ol}.pkl',
            'rb')
        content = list(pickle.load(data).values())[0]
        in_sum = np.sum(content)
        print(in_sum)
        print()

    print()
    for i in range(1, 10):
        print(f'd = 200, a = {i * 10}%')
        ol = int(200 * 0.1 * i)
        data = open(
            f'./slidshap_drifts_bi_predictions_Realworld_dataset1-timeSeries_after_filling_and_binning_windowlength200_overlap{ol}.pkl',
            'rb')
        content = list(pickle.load(data).values())[0]
        in_sum = np.sum(content)
        print(in_sum)
        print()


