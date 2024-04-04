

# input: 0s, 1s list; output: position of 1s
def getPosition(input_list):
    drift_pos = []
    for i, values in enumerate(input_list):
        if values == 1:
            drift_pos.append(i)
    return drift_pos


def ts_poslist_to_slid_poslist(pos_list, d, a):
    slid_pos = []
    for pos in pos_list:
        slid_pos.append((pos-d) / (d-d*a/100))
    return slid_pos


def ts_pos_to_slid_pos(pos, d, a):
    return (pos-d) / (d-d*a/100)


if __name__ == '__main__':
    print(ts_pos_to_slid_pos(1000, 30, 0.7))

