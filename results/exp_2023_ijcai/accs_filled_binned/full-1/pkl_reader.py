import pickle
import numpy as np
from src.drift_detection import _slidSHAPPos_to_tsPos
from src.util2 import getPosition


if __name__ == '__main__':
    f = open(
        './ts_drifts_bi_predictions_Realworld_accs_filled_binned_windowlength100_overlap70.pkl',
        'rb')
    pkl = pickle.load(f)
    ts_predictions = list(pkl.values())[0]
    print(ts_predictions)
    print(len(ts_predictions))
    print()

    print(getPosition(ts_predictions))
    print(len(getPosition(ts_predictions)))
