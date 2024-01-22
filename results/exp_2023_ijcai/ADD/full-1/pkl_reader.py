import pickle
import numpy as np
from src.drift_detection import _slidSHAPPos_to_tsPos


def _bi_to_pos0(bi_list):
    result = []
    for k, v in enumerate(bi_list):
        if v == 1:
            result.append(k)
    return result


if __name__ == '__main__':

    f1 = open(
        './full-1/slidshap_drifts_bi_predictions_Synthetic_6Concepts_ADD_5000dataPerConcept_10category_windowlength100_overlap10.pkl',
        'rb')
    f2 = open(
        './full-1/ts_drifts_bi_predictions_Synthetic_6Concepts_ADD_5000dataPerConcept_10category_windowlength100_overlap10.pkl',
        'rb')
    data1 = list(pickle.load(f1).values())[0]
    data2 = list(pickle.load(f2).values())[0]

    print(data1)
    print(_bi_to_pos0(data1))
    print(len(data1))
    print()

    print(data2)
    print(len(data2))
    print()

    result = _slidSHAPPos_to_tsPos(data1, 100, 10)
    print(result)
    print(_bi_to_pos0(result))
    print(len(result))
