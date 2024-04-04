import numpy as np
import pandas as pd
from src._loadingdata import load_data


def binning(data, intervals):

    # delete the header and row name
    data = np.delete(np.delete(data, 0, axis=0), 0, axis=1)
    data = np.array(data, dtype=float)

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

    resultArray = np.array(resultArray)
    resultArray = np.transpose(resultArray)
    return resultArray


def binning_global(data, intervals):

    # delete the header and row name
    data = np.delete(np.delete(data, 0, axis=0), 0, axis=1)
    data = np.array(data, dtype=float)
    print(f'the shape of original data {data.shape}')

    max_values = np.max(data, axis=0)
    min_values = np.min(data, axis=0)

    # Calculate overall min and max values for the entire dataset
    overall_min = np.min(min_values)
    overall_max = np.max(max_values)

    width = (overall_max - overall_min) / intervals

    resultArray = []

    for column in range(data.shape[1]):
        # Calculate bins and labels for each column based on overall min and max
        bins = [-np.inf] + [overall_min + j * width for j in range(intervals-1)] + [np.inf]
        labels = range(0, intervals)

        # Apply binning to each column
        binned_data = pd.cut(data[:, column], bins=bins, labels=labels)
        resultArray.append(binned_data.codes)

    resultArray = np.array(resultArray)
    resultArray = np.transpose(resultArray)
    return resultArray


def main_local(file, intervals):
    binned_data = binning(file, intervals)
    print(f'the shape of binned data {binned_data.shape}')
    np.savetxt('../data/' + file + '_binned50.csv', binned_data, delimiter=',', fmt='%d')


def main_global(file, intervals):
    binned_data = binning_global(file, intervals)
    print(f'the shape of binned data {binned_data.shape}')
    np.savetxt('../data/' + file + '_binned50.csv', binned_data, delimiter=',', fmt='%d')


if __name__ == '__main__':
    # main_local("Realworld_BMW.DE_minutely_events", 49)
    main_global("Realworld_Open_hourly_events", 49)







