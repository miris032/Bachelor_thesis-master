import pickle
import numpy as np


def getPosition(input_list):
    drift_pos = []
    for i, values in enumerate(input_list):
        if values == 1:
            drift_pos.append(i)
    return drift_pos


if __name__ == '__main__':
    f = open(
        './slidshap_drifts_bi_predictions_Realworld_dataset1-timeSeries_after_filling_windowlength200_overlap140.pkl',
        'rb')
    data = pickle.load(f)
    content = list(data.values())[0]
    print(content)
    print(getPosition(content))
