import numpy as np
from numpy import nan
import pandas as pd
from pathlib import Path
import os
root = Path(__file__).resolve().parent.parent


def load_data0(data):

    path = f'{root}/' + data

    if os.path.exists(path + '.pkl'):
        _mydata = pd.read_pickle(path + '.pkl')

    elif os.path.exists(path + '.txt'):
        _mydata = np.loadtxt(path + '.txt', delimiter=",")

    elif os.path.exists(path + '.csv'):
        _mydata = np.asarray(pd.read_csv(path + '.csv', delimiter=",", header=None))
    #         _mydata = _mydata)
    else:
        raise FileNotFoundError(path+'.*')

    # _mydata = np.delete(_mydata, [1, 2, 3, 4], axis=1)
    _mydata = np.array(_mydata)
    _mydata = _mydata[:, 1:]
    _mydata = np.transpose(_mydata)
    _mydata = _mydata[:, 1:]

    return _mydata


def data_fill(files):
    dataset = load_data0('biobankdata/' + files)

    dataset = dataset.astype(float)
    dataset = pd.DataFrame(dataset)
    print("dataset has N/A? " + str(dataset.isnull().values.any()))

    # print the rows with missing values
    columns_with_missing_values = dataset.columns[dataset.isnull().any()].tolist()
    # columns_with_missing_values = [x + 1 for x in columns_with_missing_values]
    print("Columns with missing values: ", columns_with_missing_values)

    # Filling with median
    # accs filling
    # dataset = dataset.fillna(dataset.median())

    # single dataset filling
    dataset = dataset.apply(lambda row: row.fillna(row.median()) if row.name == 0 else row.fillna(0), axis=1)

    # accs filling
    # dataset = dataset.apply(lambda row: row.fillna(row.median()), axis=1)

    print("dataset has N/A? " + str(dataset.isnull().values.any()))

    dataset = np.array(dataset)
    dataset = np.transpose(dataset)
    np.savetxt(f'{files}_filled.csv', dataset, delimiter=", ")

    dataset = pd.DataFrame(dataset)
    print(dataset.median())

    '''columns_with_missing_values = np.array(columns_with_missing_values)
    columns_with_missing_values = np.transpose(columns_with_missing_values)
    np.savetxt('accs_missing_values_rows.csv', columns_with_missing_values, delimiter=", ", fmt='%d')'''


if __name__ == '__main__':
    data_fill('dataset9-timeSeries')




