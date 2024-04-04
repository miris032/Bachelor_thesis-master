import pickle

import pandas as pd

from src.util2 import getPosition

if __name__ == '__main__':
    pkl = pd.read_pickle(
        f'ts_drifts_bi_predictions_Realworld_Open_hourly(events)_binned_windowlength30_overlap21.pkl')
    ts_predictions = list(pkl.values())[0]
    print(ts_predictions)
    print(getPosition(ts_predictions))
