import random
import sys

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
root = Path(__file__).resolve().parent.parent


def random_walk(initial_price, num_steps, volatilities):
    prices = [initial_price]
    for i in range(num_steps):
        drift = 0  # 漂移项，可以根据需要添加
        shock = np.random.normal(loc=0, scale=volatilities[i % len(volatilities)])  # 随机波动项，代表价格的波动
        price = prices[-1] + drift + shock
        prices.append(price)
    return prices


def random_triangular(initial_price, num_steps, volatilities):
    prices = [initial_price]
    num_volatilities = len(volatilities)
    for i in range(num_steps):
        drift = 0  # 漂移项，可以根据需要添加
        volatility_index = i % num_volatilities  # 循环使用波动率数组中的元素

        # 生成三角函数分布的随机数
        min_val = -volatilities[volatility_index]
        max_val = volatilities[volatility_index]
        mode_val = 0  # 三角函数分布的峰值可以根据需要调整
        shock = np.random.triangular(left=min_val, mode=mode_val, right=max_val)

        price = prices[-1] + drift + shock
        prices.append(price)
    return prices


def biobankdata_insertion(file, insert_pos, row_num):

    if file.startswith('dataset'):
        dataset_1_to_9 = True
    elif file.startswith('accs'):
        dataset_1_to_9 = False
    else:
        sys.exit('Unexpected data')

    df = pd.read_csv(f'{root}/data/biobankdata/{file}.csv', delimiter=",", header=None)
    date = df.iloc[1:, 0].tolist()
    header = df.iloc[0, :].tolist()
    data = np.asarray(df.iloc[1:, 1:].T.astype(float))
    print(data.shape)

    column_num = data.shape[0]
    insert_pos_after = []
    seed = 42

    for pos in insert_pos:

        np.random.seed(seed)
        pos += insert_pos.index(pos) * row_num
        insert_pos_after.append(pos)
        tmp_arr = np.empty((column_num, row_num))

        for i in range(column_num):
            if not dataset_1_to_9 or (dataset_1_to_9 and i == 0):
                np.random.seed(seed)
                min_value_30th = np.sort(data[i][pos-1000: pos])[10]
                max_value_30th = np.sort(data[i][pos-1000: pos])[-10]

                '''mean = (min_value_30th + max_value_30th) / 2 - 30
                std_dev = (max_value_30th - min_value_30th) / 2
                std_dev *= 2'''

                y = np.empty((0,))
                for _ in range(int(row_num/10)):
                    mean = max_value_30th / random.uniform(2, 10)

                    # 生成0到1之间的随机数
                    random_prob = random.random()

                    # 根据随机数的值来决定增大或减小std_dev
                    if random_prob < 0.8:  # 80%的概率
                        std_dev = max_value_30th / random.uniform(15, 20)
                    else:  # 20%的概率
                        std_dev = max_value_30th / random.uniform(1, 3)

                    y1 = np.abs(np.random.normal(mean, std_dev, 10))
                    y = np.hstack((y, y1))

                '''num_large_values = int(row_num * 0.01)  # 偶尔出现较大值的数量
                large_values_indices = np.random.choice(np.arange(row_num), size=num_large_values, replace=False)  # 随机选择插入位置
                large_values = np.random.uniform(0, max_value_30th*4, size=num_large_values)  # 在指定范围内生成均匀分布的大值
                y[large_values_indices] = large_values  # 将大值替换掉原始数据的一部分'''

            elif dataset_1_to_9 and i == 5:
                y = np.mean(data[5])
            elif dataset_1_to_9 and i == 1:
                y = [1 if i < row_num/5 else 0 for i in range(row_num)]
            elif dataset_1_to_9 and i == 2:
                y = [0 if i < row_num*2/5 else 1 if i < row_num*3/5 else 0 for i in range(row_num)]
            elif dataset_1_to_9 and i == 3:
                y = [0 if i < row_num*3/5 else 1 if i < row_num*4/5 else 0 for i in range(row_num)]
            elif dataset_1_to_9 and i == 4:
                y = [1 if i > row_num*4/5 else 0 for i in range(row_num)]
            tmp_arr[i, :] = y

        data = np.insert(data, pos, tmp_arr.transpose(), axis=1)

    data = data.transpose()
    if not dataset_1_to_9:
        np.savetxt('accs(events).csv', data, delimiter=",", fmt='%s')
    else:
        np.savetxt('dataset1-timeSeries(events).csv', data, delimiter=",", fmt='%s')

    # 绘制折线图
    plt.figure(figsize=(15, 3))
    for i in range(column_num):
        plt.plot(data[:, i], label=f'Column {i + 1}', linewidth=0.2, alpha=0.5)

    # 使用 fill_between 函数填充插入数据的区域
    ylim = plt.gca().get_ylim()
    for pos in insert_pos_after:
        print(f'{pos + 1} - {pos + row_num}')
        plt.fill_between(range(pos + 1, pos + row_num + 1), ylim[0], ylim[1], color='gray', hatch='///', alpha=0.1)

    plt.title('Data Line Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def stockdata_insertion(file, insert_pos, row_num):

    if any(file.startswith(prefix) for prefix in ('Open', 'Close', 'High', 'Low', 'Volume', 'Adj Close')):
        mix = True
    else:
        mix = False

    # mix
    if mix:
        df = pd.read_csv(f'{root}/data/ticker_data/{file}.csv', delimiter=",", header=None)
    # single stock
    else:
        time_category = file.split('_')[1]
        file_name = file.split('_')[0]
        df = pd.read_csv(f'{root}/data/ticker_data/{time_category}/{file_name}.csv', delimiter=",", header=None)

    date = df.iloc[1:, 0].tolist()
    header = df.iloc[0, :].tolist()
    data = np.asarray(df.iloc[1:, 1:].T.astype(float))

    column_num = data.shape[0]
    insert_pos_after = []
    seed = 42

    for pos in insert_pos:
        #np.random.seed(pos)
        pos += insert_pos.index(pos) * row_num
        insert_pos_after.append(pos)
        tmp_arr = np.empty((column_num, row_num))
        for i in range(column_num):
            np.random.seed(seed+i)

            # mix
            if mix:
                if i == 6 or i == 7 or i == 8:
                    y = random_walk(data[i][pos], row_num - 1, (0.05, 0.075))
                elif i == 0 or i == 4 or i == 9:
                    y = random_walk(data[i][pos], row_num - 1, (0.2, 0.4, 0.6))
                else:
                    y = random_walk(data[i][pos], row_num - 1, (0.9, 1.2, 1.5))
            # single
            else:
                if time_category == 'daily':
                    if i < 5:
                        y = random_walk(data[i][pos], row_num-1, 0.5)
                    else:
                        y = np.random.uniform(low=np.sort(data[i][pos-400: pos])[30], high=np.sort(data[i][pos-400: pos])[-30],
                                              size=row_num)
                elif time_category == 'hourly':
                    if i < 4:
                        y = random_walk(data[i][pos], row_num-1, (0.2, 0.3, 0.4))
                    else:
                        y = np.random.uniform(low=np.sort(data[i][pos-400: pos])[30], high=np.sort(data[i][pos-400: pos])[-30],
                                              size=row_num)
                    y = [element + 5 for element in y]
                    y = np.round(y, 5)
                # minutely
                else:
                    if i < 4:
                        y = random_triangular(data[i][pos], row_num-1, (0.2, 0.3, 0.4))
                    else:
                        y = np.random.uniform(low=np.sort(data[i][pos-100: pos])[5], high=np.sort(data[i][pos-100: pos])[-5],
                                              size=row_num)
            tmp_arr[i, :] = y
        data = np.insert(data, pos, tmp_arr.transpose(), axis=1)
        seed += 1

    data = data.transpose()
    if mix:
        np.savetxt(f'Open_hourly(events).csv', data, delimiter=",", fmt='%s')
    else:
        np.savetxt(f'BMW.DE(events)_{time_category}.csv', data, delimiter=",", fmt='%s')

    # 绘制折线图
    plt.figure(figsize=(15, 3))
    for i in range(column_num):
        plt.plot(data[:, i], label=f'Column {i + 1}', linewidth=0.2, alpha=0.5)

    # 使用 fill_between 函数填充插入数据的区域
    ylim = plt.gca().get_ylim()
    for pos in insert_pos_after:
        print(f'{pos + 1} - {pos + row_num}')
        plt.fill_between(range(pos + 1, pos + row_num + 1), ylim[0], ylim[1], color='gray', hatch='///', alpha=0.1)

    plt.title('Data Line Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    biobankdata_insertion('dataset1-timeSeries', (2500, 6000, 9500, 13000, 16500), 1500)
    # biobankdata_insertion('accs_filled', (1500, 5000, 8500, 12000, 15500), 1000)


    # stockdata_insertion('BMW.DE_hourly', (1000, 2000, 3000, 4000, 5000), 500)
    # stockdata_insertion('BMW.DE_minutely', (300, 600, 900, 1200, 1500), 200)

    # stockdata_insertion('Open_hourly', (1000, 2000, 3000, 4000, 5000), 500)
