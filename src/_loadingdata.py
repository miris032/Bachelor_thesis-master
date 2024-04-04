import pandas as pd
import os
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent.parent


def load_data(data):

    path = f'{root}/data/' + data

    if os.path.exists(path + '.pkl'):
        _mydata = pd.read_pickle(path + '.pkl')

    elif os.path.exists(path + '.txt'):
        _mydata = np.loadtxt(path + '.txt', delimiter=",")

    elif os.path.exists(path + '.csv'):
        _mydata = np.asarray(pd.read_csv(path + '.csv', delimiter=",", header=None))
    #         _mydata = _mydata)
    else:
        raise FileNotFoundError(path+'.*')


    ########################### 删除第1, 2列的时间（对于不同的数据需要调整参数） ###########################
    # _mydata = np.delete(_mydata, [0, 1], axis=1)


    return _mydata


def load_data_CD():
    _data = 'concept_drifted_data'
    _mydata = load_data(_data)

    ran = np.arange(0.05, 1.05, 0.05)

    col_to_drop = []
    for i in range(len(_mydata.columns)):
        if len(_mydata[_mydata.columns[i]].unique()) == 10000:
            name_col = _mydata.columns[i]
            x = _mydata[name_col]
            binned = np.digitize(x, ran)
            col_to_drop.append(name_col)
            if i % 2 == 1:
                _mydata[f'{name_col}_binned'] = binned

    print(col_to_drop)

    _mydata = _mydata.drop(col_to_drop, axis=1).to_numpy()

    return _mydata


if __name__ == '__main__':
    print(load_data('Open_test'))
    print(len(load_data('Open_test')))
