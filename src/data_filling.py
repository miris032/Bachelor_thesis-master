import numpy as np
from numpy import nan
import pandas as pd
from pathlib import Path
import os
root = Path(__file__).resolve().parent.parent


def single_dataset_filling(df):
    date = df.iloc[1:, 0].tolist()
    header = df.iloc[0, :].tolist()
    dataset = df.iloc[1:, 1:].T.astype(float)

    print('data filling: ')
    print("dataset has N/A before filling? " + str(dataset.isnull().values.any()))

    # print the rows with missing values
    columns_with_missing_values = dataset.columns[dataset.isnull().any()].tolist()
    # columns_with_missing_values = [x + 1 for x in columns_with_missing_values]
    print("Columns with missing values: ", columns_with_missing_values)

    # single dataset filling
    dataset = dataset.apply(lambda row: row.fillna(row.median()) if row.name == 0 else row.fillna(0), axis=1)

    print("dataset has N/A after filling? " + str(dataset.isnull().values.any()))
    print()

    dataset = dataset.T.astype(str)
    dataset = np.array(dataset)
    dataset = np.insert(dataset, 0, values=date, axis=1)
    dataset = np.insert(dataset, 0, values=header, axis=0)

    return dataset


def accs_filling(df):
    date = df.iloc[1:, 0].tolist()
    header = df.iloc[0, :].tolist()
    dataset = df.iloc[1:, 1:].T.astype(float)

    print('data filling: ')
    print("dataset has N/A before filling? " + str(dataset.isnull().values.any()))

    # print the rows with missing values
    columns_with_missing_values = dataset.columns[dataset.isnull().any()].tolist()
    # columns_with_missing_values = [x + 1 for x in columns_with_missing_values]
    print("Columns with missing values: ", columns_with_missing_values)

    # accs filling
    dataset = dataset.fillna(dataset.median())
    dataset = dataset.apply(lambda row: row.fillna(row.median()), axis=1)

    print("dataset has N/A after filling? " + str(dataset.isnull().values.any()))
    print()

    dataset = dataset.T.astype(str)
    dataset = np.array(dataset)
    dataset = np.insert(dataset, 0, values=date, axis=1)
    dataset = np.insert(dataset, 0, values=header, axis=0)

    return dataset
