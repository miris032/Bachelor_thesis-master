import pickle
import numpy as np
from src.drift_detection import _slidSHAPPos_to_tsPos

if __name__ == '__main__':
    f1 = open(
        './slidshap_drifts_bi_predictions_Realworld_1COV.DE_minutely_oneweek_windowlength200_overlap140.pkl',
        'rb')

    data1 = list(pickle.load(f1).values())[0]

    print(data1)


