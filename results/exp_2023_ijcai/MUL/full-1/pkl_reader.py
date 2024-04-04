import pickle
import numpy as np

if __name__ == '__main__':
    f = open(
        './slidshap_drifts_bi_predictions_Synthetic_6Concepts_MUL_5000dataPerConcept_4category_windowlength100_overlap70.pkl',
        'rb')
    data = pickle.load(f)
    print(data)
