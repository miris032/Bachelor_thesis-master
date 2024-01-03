import numpy as np
import pandas as pd
from src._loadingdata import load_data



def binning(input_file, intervals):

    data = load_data(input_file)
    # delete the header and row name
    # data = np.delete(np.delete(data, 0, axis=0), 0, axis=1)
    data = np.array(data, dtype=float)
    print(f'the shape of original data {data.shape}')

    max_values = np.max(data, axis=0)
    min_values = np.min(data, axis=0)

    resultArray = []
    for column in range(data.shape[1]):

        # If there are l numbers in this column, and l < intervals from input, we cannot use the intervals as lables.
        l = len(set(data[:, column]))
        if l < intervals:
            bins = [-np.inf] \
                   + [min_values[column] + j * width for j in range(l - 1)] \
                   + [np.inf]
            labels = range(0, l)


        else:
            width = (max_values[column] - min_values[column]) / intervals
            bins = [-np.inf] \
                   + [min_values[column] + j * width for j in range(intervals-1)] \
                   + [np.inf]

            '''print(f"Column {column + 1}: Min = {min_values[column]}, Max = {max_values[column]}")
            print(bins)
            print()'''

            labels = range(0, intervals)


        # print(data[:, column])
        binned_data = pd.cut(data[:,column], bins=bins, labels=labels)
        resultArray.append(binned_data.codes)

        # print(binned_data)
        # print()
        # print()
    resultArray = np.array(resultArray)
    resultArray = np.transpose(resultArray)
    return resultArray



'''def bin_2d_array(data, bin_size):

    data = load_data(data)
    if bin_size <= 0:
        raise ValueError("分箱单位必须大于0")

    num_rows, num_cols = data.shape
    bin_data = np.zeros((num_rows, num_cols), dtype=int)

    for i in range(num_rows):
        for j in range(num_cols):
            bin_data[i, j] = data[i, j] // bin_size

    return bin_data'''


def main(file, intervals):
    binned_data = binning(file, intervals)
    print(f'the shape of binned data {binned_data.shape}')
    np.savetxt('../data/' + file + '_binned.csv', binned_data, delimiter=',')



if __name__ == '__main__':
    main("Realworld_High_hourly", 19)






