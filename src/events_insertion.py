import random
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
root = Path(__file__).resolve().parent.parent


def random_walk(initial_price, end_price, num_steps, volatilises):
    # np.random.seed(42)  # 设置随机数种子
    prices = [initial_price]
    for i in range(num_steps):
        drift = (end_price - initial_price) / 210  # 漂移项，可以根据需要添加
        # shock = np.random.normal(loc=0, scale=volatilises[i % len(volatilises)])
        shock = np.random.normal(loc=0, scale=volatilises[i % len(volatilises)])  # 随机波动项，代表价格的波动
        price = prices[-1] + drift + shock
        prices.append(price)
    return prices[1:]


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


def random_sin(initial_point, num_cycles, num_points_per_cycle):
    total_points = num_cycles * num_points_per_cycle
    x = np.linspace(100, 2 * np.pi * num_cycles, total_points)
    y = (np.sin(x)).astype(int) + np.random.uniform(1.5, 3.5, total_points).round(3)
    y = [float(point + initial_point) for point in y]
    return y


def gaussian_drift_generation(initial_point, number, l, nu):
    kernel = Matern(length_scale=l, nu=nu)
    gp = GaussianProcessRegressor(kernel=kernel)
    y = gp.sample_y(np.linspace(0, 10, number)[:, np.newaxis])
    y = [float(point + initial_point) for point in y]
    y = [point * 1.5 for point in y]
    return y


def random_biobank(row_num, max_value_30th):
    y = np.empty((0,))
    for _ in range(int(row_num / 10)):
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
    return y


def biobankdata_insertion(file, insert_pos, row_num, event_type):

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
    seed = 42

    for pos_index, pos in enumerate(insert_pos):
        for column in range(column_num):
            np.random.seed(seed)
            max_value_30th = np.sort(data[column][pos - 1000: pos])[-10]

            # single
            if dataset_1_to_9:
                # acc column
                if column == 0:
                    # sudden
                    if event_type == 'sudden':
                        y = random_sin(250, 25, 30)
                    # incremental
                    else:
                        y = np.empty((0,))
                        for _ in range(int(row_num / 10)):
                            min_value_30th = np.sort(data[column][pos - 1000: pos])[10]
                            max_value_30th = np.sort(data[column][pos - 1000: pos])[-10]
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

                # MET column
                elif column == 5:
                    # sudden
                    if event_type == 'sudden':
                        y = np.mean(data[5] + random.uniform(5, 10))
                    # incremental
                    else:
                        y = np.mean(data[5])

                # classification columns
                elif column == 1:
                    y = [1 for i in range(row_num)]
                elif column == 2:
                    y = [0 for i in range(row_num)]
                elif column == 3:
                    y = [0 for i in range(row_num)]
                elif column == 4:
                    y = [0 for i in range(row_num)]

            # accs
            else:
                # sudden
                if event_type == 'sudden':
                    data[column][pos: pos + row_num] = random_biobank(row_num, max_value_30th)

                # incremental
                else:
                    change_pos = 80 * column
                    y1 = data[column][pos: pos + change_pos].tolist()
                    y2 = random_biobank((row_num-change_pos), max_value_30th).tolist()
                    print(f'y1 = data[{column}][{pos}: {pos} + {change_pos}].tolist()')
                    print(f'y2 = random_biobank(({row_num}-{change_pos}), {max_value_30th}).tolist()')
                    print()
                    data[column][pos: pos + row_num] = y1 + y2

    data = data.transpose()
    if not dataset_1_to_9:
        if event_type == 'sudden':
            np.savetxt('../data/biobankdata/accs(events2).csv', data, delimiter=",", fmt='%s')
        else:
            np.savetxt('../data/biobankdata/accs(events).csv', data, delimiter=",", fmt='%s')
    else:
        if event_type == 'sudden':
            np.savetxt('../data/biobankdata/dataset1-timeSeries(events2).csv', data, delimiter=",", fmt='%s')
        else:
            np.savetxt('../data/biobankdata/dataset1-timeSeries(events).csv', data, delimiter=",", fmt='%s')

    # 绘制折线图
    plt.figure(figsize=(15, 3))
    for column in range(column_num):
        plt.plot(data[:, column], label=f'Column {column + 1}', linewidth=0.2, alpha=0.5)

    # 使用 fill_between 函数填充插入数据的区域
    ylim = plt.gca().get_ylim()
    for pos in insert_pos:
        print(f'{pos + 1} - {pos + row_num}')
        plt.fill_between(range(pos + 1, pos + row_num + 1), ylim[0], ylim[1], color='gray', hatch='///', alpha=0.1)

    plt.title('Data Line Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def stockdata_insertion(file, insert_pos, row_num, event_type):

    if any(file.startswith(prefix) for prefix in ('Open', 'Close', 'High', 'Low', 'Volume', 'Adj Close')):
        mix = True
        df = pd.read_csv(f'{root}/data/ticker_data/{file}.csv', delimiter=",", header=None)
    else:
        mix = False
        time_category = file.split('_')[1]
        file_name = file.split('_')[0]
        df = pd.read_csv(f'{root}/data/ticker_data/{time_category}/{file_name}.csv', delimiter=",", header=None)

    date = df.iloc[1:, 0].tolist()
    header = df.iloc[0, :].tolist()
    data = np.asarray(df.iloc[1:, 1:].T.astype(float))

    column_num = data.shape[0]
    seed = 42

    for pos_index, pos in enumerate(insert_pos):
        #np.random.seed(pos)
        '''pos += insert_pos.index(pos) * row_num
        insert_pos_after.append(pos)
        tmp_arr = np.empty((column_num, row_num))'''
        for column in range(column_num):
            #np.random.seed(seed+column)
            np.random.seed(42 + pos_index)  # 设置随机数种子

            # mix
            if mix:
                if column == 6 or column == 7 or column == 8:
                    v = (0.05, 0.075)
                elif column == 0 or column == 4 or column == 9:
                    v = (0.2, 0.4, 0.6)
                else:
                    v = (0.9, 1.2, 1.5)

                # sudden
                if event_type == 'sudden':
                    data[column][pos: pos + row_num] = random_walk(data[column][pos], data[column][pos + row_num], row_num, v)
                # incremental
                else:
                    '''y = random_walk(data[1][pos], row_num - 1, (0.9, 1.2, 1.5))
                    y = [point + 250 for point in y]'''
                    # y = random_sin(400, 4, 50)
                    change_pos = 20 * column
                    y1 = data[column][pos: pos + change_pos].tolist()
                    y2 = random_walk(data[column][pos + change_pos], data[column][pos + row_num], (row_num - change_pos), v)
                    data[column][pos: pos + row_num] = y1 + y2


            # single
            else:
                if time_category == 'daily' or time_category == 'hourly':
                    v = (0.75, 1, 1.25)
                # minutely
                else:
                    v = (0.02, 0.03, 0.04)

                # sudden
                if event_type == 'sudden':
                    data[column][pos: pos+row_num] = random_walk(data[column][pos], data[column][pos + row_num], row_num, v)
                # incremental
                else:
                    change_pos = int(column * row_num / (len(insert_pos) - 1))
                    y1 = data[column][pos: pos + change_pos].tolist()
                    y2 = random_walk(data[column][pos + change_pos], data[column][pos + row_num], row_num - change_pos, v)
                    data[column][pos: pos+row_num] = y1 + y2

    data = data.transpose()
    if mix:
        if event_type == 'sudden':
            np.savetxt('../data/ticker_data/Open_hourly(events2).csv', data, delimiter=",", fmt='%s')
        else:
            np.savetxt('../data/ticker_data/Open_hourly(events).csv', data, delimiter=",", fmt='%s')

    else:
        if event_type == 'sudden':
            np.savetxt(f'../data/ticker_data/{time_category}/BMW.DE(events2).csv', data, delimiter=",", fmt='%s')
        else:
            np.savetxt(f'../data/ticker_data/{time_category}/BMW.DE(events).csv', data, delimiter=",", fmt='%s')


    # 绘制折线图
    plt.figure(figsize=(15, 3))
    for column in range(column_num):
        plt.plot(data[:, column], label=f'Column {column + 1}', linewidth=0.2, alpha=0.5)

    # 使用 fill_between 函数填充插入数据的区域
    ylim = plt.gca().get_ylim()
    for pos in insert_pos:
        print(f'{pos + 1} - {pos + row_num}')
        plt.fill_between(range(pos + 1, pos + row_num + 1), ylim[0], ylim[1], color='gray', hatch='///', alpha=0.1)

    plt.title('Data Line Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # biobankdata_insertion('dataset1-timeSeries_filled', (2500, 6000, 9500, 13000, 16500), 750, 'incremental') # incremental
    # biobankdata_insertion('accs_filled', (3000, 6000, 9000, 12000, 15000), 750, 'sudden') # incremental


    # stockdata_insertion('BMW.DE_daily', (1200, 2400, 3600, 4800, 6000), 250, 'incremental')
    # stockdata_insertion('BMW.DE_hourly', (1000, 2000, 3000, 4000, 5000), 200, 'incremental')
    stockdata_insertion('BMW.DE_minutely', (300, 600, 900, 1200, 1500), 70, 'incremental')
    # stockdata_insertion('Open_hourly', (1000, 2000, 3000, 4000, 5000), 200, 'sudden') # incremental
