import matplotlib.pyplot as plt
import numpy as np
from src._loadingdata import load_data
from enum import Enum


class Color(Enum):
    LIGHT_ORANGE = '#FFDDBB'
    LIGHT_PURPLE = '#E6E6FA'
    LIGHT_GREEN = '#B0EAB0'
    LIGHT_RED = '#FFCCCC'


# 1: light, 2: moderate-vigorous, 3: sedentary, 4: sleep
def get_color(color_code):
    if color_code == 1:
        return Color.LIGHT_ORANGE.value
    elif color_code == 2:
        return Color.LIGHT_GREEN.value
    elif color_code == 3:
        return Color.LIGHT_RED.value
    elif color_code == 4:
        return Color.LIGHT_PURPLE.value
    else:
        return None


def find_row_with_most_ones(inputdata):
    max_ones = 0
    max_ones_row = -1

    for i in range(1, inputdata.shape[1]):
        count_ones = np.count_nonzero(inputdata[:, i] == 1)
        if count_ones > max_ones:
            max_ones = count_ones
            max_ones_row = i

    return max_ones_row


def drawPlot(data_Nr, d, a_list):
    org_data = load_data(f"dataSets/dataset{data_Nr}-timeSeries_new")
    l = len(a_list)
    fig, axs = plt.subplots(l, 1)

    # draw subPlots with different a
    for i in range(l):
        shapleyed_data = load_data(f"results/{data_Nr})/{d}, {a_list[i]}%")

        # 1. draw the line plot
        acc = shapleyed_data[0, :]
        axs[i].plot(acc, label=f'acc', linewidth=0.3)
        axs[i].set_title(f"d: {d},  a: {a_list[i]}")

        # 2. filling color on the line plot
        a = int(int(a_list[i]) * d * 0.01)
        current_color = None
        start_index = None
        for j in range(shapleyed_data.shape[1]):
            # find corresponding interval back on time-series
            x_start = j * (d - a)
            x_end = (j + 1) * d - j * a

            movementTyp = find_row_with_most_ones(org_data[x_start:x_end])
            if current_color is None:
                current_color = movementTyp
                start_index = j

            if movementTyp != current_color:
                # filling color
                color = get_color(current_color)
                axs[i].fill_betweenx([0, 1], start_index, j, color=color, alpha=1)

                # update the starting index of the current fill color and area
                current_color = movementTyp
                start_index = j

        # filling the last area
        '''color = get_color(current_color)
        axs[i].fill_betweenx([0, 1], start_index, j + 1, color=color, alpha=1)'''


        '''for j in range(shapleyed_data.shape[1]):

            # find corresponding interval back on time-series
            x_start = j * (50-5)
            x_end = (j+1)*50 - j*5

            # filling corresponding color
            movementTyp = find_row_with_most_ones(org_data[x_start:x_end])
            if movementTyp == 1:
                axs[i].fill_betweenx([0, 1], j, j, color='lightblue', alpha=1)
            elif movementTyp == 2:
                axs[i].fill_betweenx([0, 1], j, j, color='lightyellow', alpha=1)
            elif movementTyp == 3:
                axs[i].fill_betweenx([0, 1], j, j, color='lightgreen', alpha=1)
            elif movementTyp == 4:
                axs[i].fill_betweenx([0, 1], j, j, color='mistyrose', alpha=1)'''

    # legend
    legend_labels = ['light', 'MVPA', 'sedentary', 'sleep']
    legend_colors = [Color.LIGHT_ORANGE.value, Color.LIGHT_GREEN.value, Color.LIGHT_RED.value, Color.LIGHT_PURPLE.value]
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in
                      legend_colors]
    plt.legend(legend_handles, legend_labels, loc=2, bbox_to_anchor=(1.0, 1.0))

    fig.suptitle(f"dataset{data_Nr}-timeSeries_new.csv{org_data.shape} \n20 bins")
    fig.set_size_inches(16, 8)
    plt.subplots_adjust(hspace=0.8)
    plt.show()


def draw_line_separately(data_Nr, d, a):
    fig, axs = plt.subplots(6, 1)

    org_data = load_data(f"dataSets/dataset{data_Nr}-timeSeries_new")
    shapleyed_data = load_data(f"results/{data_Nr})/{d}, {a}%")
    a = int(int(a) * d * 0.01)

    header = ('shapley value for acc', 'shapley value for light', 'shapley value for MVPA',
              'shapley value for sedentary', 'shapley value for sleep', 'shapley value for MET')
    for i in range(6):
        # 1. draw subPlots with different line
        axs[i].plot(shapleyed_data[i, :], label=header[i], linewidth=0.3)
        axs[i].set_title(header[i])

        # 2. filling color on the line plot
        current_color = None
        start_index = None
        for j in range(shapleyed_data.shape[1]):
            # find corresponding interval back on time-series
            x_start = j * (d - a)
            x_end = (j + 1) * d - j * a

            movementTyp = find_row_with_most_ones(org_data[x_start:x_end])
            if current_color is None:
                current_color = movementTyp
                start_index = j

            if movementTyp != current_color:
                # filling color
                color = get_color(current_color)
                axs[i].fill_betweenx([0, 1], start_index, j, color=color, alpha=1)

                # update the starting index of the current fill color and area
                current_color = movementTyp
                start_index = j

        # filling the last area
        '''color = get_color(current_color)
        axs[i].fill_betweenx([0, 1], start_index, j + 1, color=color, alpha=1)'''

    legend_labels = ['light', 'MVPA', 'sedentary', 'sleep']
    legend_colors = [Color.LIGHT_ORANGE.value, Color.LIGHT_GREEN.value, Color.LIGHT_RED.value, Color.LIGHT_PURPLE.value]
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in
                      legend_colors]
    plt.legend(legend_handles, legend_labels, loc=2, bbox_to_anchor=(1.0, 1.0))

    fig.suptitle(f"dataset{data_Nr}-timeSeries_new.csv{org_data.shape} \nlength d = {d}, overlap a = {a}, bins = 20")
    fig.set_size_inches(16, 8)
    plt.subplots_adjust(hspace=0.8)
    plt.show()


if __name__ == '__main__':
    # drawPlot(9, 50, (10, 50))
    draw_line_separately(1, 100, 10)

