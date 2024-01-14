import numpy as np
import pandas as pd


def get_features(file, selected_rows_range, feature_columns):
    # 读取CSV文件，确保跳过第一行作为列名
    data = pd.read_csv(f'{file}.csv', delimiter=",", header=None, skiprows=1).values

    # 选择特定的列
    data1 = data[:, [1, 2, 3, 4, 5]]

    # 创建DataFrame
    data1 = pd.DataFrame(data1)

    # 选择指定的行和特征列
    selected_rows = list(range(selected_rows_range.start, selected_rows_range.stop + 1))
    selected_data = data1.iloc[selected_rows, feature_columns]

    # 对特征列进行汇总（计数）
    feature_counts = selected_data.sum(axis=0)

    return feature_counts


def run(begin, end):
    for i in range(1, 10):
        if i == 1:
            print(f'acc{i}')
            print(get_features(f'dataset{i}-timeSeries_after_filling', range(begin, end), list(range(1, 5))))
        else:
            print(f'acc{i}')
            print(get_features(f'dataset{i}-timeSeries_filled', range(begin, end), list(range(1, 5))))

if __name__ == '__main__':
    run(1752, 1811)
    # run(2537, 2596)
    # run(4409, 4468)
    # run(5375, 5434)
    # run(8334, 8393)
    # run(8696, 8755)

    # run(10206, 10265)
    # run(11232, 11292)
    # run(13165, 13224)
    # run(14191, 14251)
    # run(16123, 16183)
    # run(17271, 17330)


