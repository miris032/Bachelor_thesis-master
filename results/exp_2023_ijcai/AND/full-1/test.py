import pickle
import numpy as np

if __name__ == '__main__':
    f = open(
        './ts_drifts_bi_predictions_Synthetic_6Concepts_AND_5000dataPerConcept_10category_windowlength100_overlap80.pkl',
        'rb')
    data = pickle.load(f)
    print(data)