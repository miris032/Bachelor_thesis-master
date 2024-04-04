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
    print(selected_data)

    # 对特征列进行汇总（计数）
    feature_counts = selected_data.sum(axis=0)

    return feature_counts


def run(end):
    for i in range(1, 10):
        print(f'acc{i}')
        print(get_features(f'dataset{i}-timeSeries_filled', range(end-60, end), list(range(1, 5))))
        print()


if __name__ == '__main__':

    #run(1359)
    #run(1989)
    #run(2529)
    #run(3309)
    #run(4029)
    #run(4809)
    #run(5649)
    #run(5979)
    #run(7059)
    
    #run(7659)
    #run(7899)
    #run(8049)
    #run(8349)
    #run(9879)
    #run(10509)
    #run(10839)
    #run(11559)
    #run(12069)

    #run(12789)
    #run(14259)
    #run(15729)
    #run(16209)
    #run(16449)
    #run(17229)
    #run(17859)
    run(18429)



