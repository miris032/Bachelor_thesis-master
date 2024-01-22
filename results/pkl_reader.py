import pickle
import numpy as np

if __name__ == '__main__':
    f = open(
        './slidshap_drifts_bi_predictions_Realworld_Open_windowlength100_overlap50.pkl',
        'rb')
    data = pickle.load(f)
    print(data)
