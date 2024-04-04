import pandas as pd

import __shapley_calculation as sv
import os
from tqdm import tqdm
import numpy as np
import time
from src.binning import binning
from pathlib import Path
root = Path(__file__).resolve().parent.parent


def compute_shapleyvalues(_mydata, _type, _subsets_bound=-1, approx='max'):

    if _type == 'bounded':
        SVC = sv.ShapleyValueCalculator("total_correlation", "subsets_bounded", _subsets_bound)
    elif _type == 'full':
        SVC = sv.ShapleyValueCalculator("total_correlation", "subsets")
    shapleys = SVC.calculate_SVs(_mydata)

    return shapleys


def run_slidshaps(_mydata, _window_width, _overlap, _type, _subsets_bound, _approx):

    T_in = 0
    T_fin = _window_width

    data_current = _mydata[T_in:T_fin]

    data_shaps = np.asarray(compute_shapleyvalues(data_current.T, _type, _subsets_bound, _approx)).reshape(-1,1)

    start_time = time.time()

    for T_in in tqdm(range(_overlap, np.shape(_mydata.T)[1], _window_width - _overlap)):
        T_fin = T_in + _window_width
        data_current = _mydata[T_in:T_fin]
        shapleyvalues = np.asarray(compute_shapleyvalues(data_current.T, _type, _subsets_bound, _approx)).reshape(-1,1)
        data_shaps = np.concatenate([data_shaps, shapleyvalues], axis=1)
    running_time = time.time() - start_time

    return data_shaps, running_time








def generate(data, d, a):

    # d: window length, a: overlap in procent, aa: actual overlap
    aa = int((d/100)*a)

    print(f'{d}, {a}%. {os.path.basename(data)}')
    data = np.asarray(pd.read_csv(f'{root}/data/ticker_data/columns_hourly/' + f'{data}' + '.csv', delimiter=",", header=None))

    print(data)

    data_shaps = run_slidshaps(data, d, aa, "full", -1, 'max')
    # data_shaps = run_slidshaps(bin_2d_array(data, 10), d, aa, "full", -1, 'max')

    print(f'shaps: {data_shaps[0].shape}')


    file_name = f'{d}, {a}%.txt'
    save_path = '../data/ticker_data_result/'
    np.savetxt(save_path + file_name, data_shaps[0], delimiter=", ")
    print(data_shaps[1])
    print()

    # print(len(np.unique(data[:,0])))


def generate_4_plots(data):

    # d: window length, a: overlap in procent
    generate(data, 50, 30)
    generate(data, 50, 70)
    generate(data, 200, 30)
    generate(data, 200, 70)

    folder_path = '../data/ticker_data_result/' + os.path.basename(data)
    os.mkdir(folder_path)


if __name__ == '__main__':

    # generate_4_plots('Volume')
    generate_4_plots('High')
















