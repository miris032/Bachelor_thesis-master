import pandas as pd
from pathlib import Path
from src.util2 import getPosition

root = Path(__file__).resolve().parent.parent
from src import evaluator


def is_in_synthetic_drift_areas(point, synthetic_drift_areas):
    for area in synthetic_drift_areas:
        if area[0] <= point <= area[1]:
            return True
    return False


def reduce_FP(located_points_without_synthetic_drifts, located_points, actual_points, interval, l):

    synthetic_drift_areas =[]
    for synthetic_drift in actual_points:
        synthetic_drift_areas.append((synthetic_drift, synthetic_drift + interval))

    located_points_new = []
    for point in located_points:

        if (not any(abs(point - value) <= l for value in located_points_without_synthetic_drifts)) \
                or is_in_synthetic_drift_areas(point, synthetic_drift_areas):
            located_points_new.append(point)
    return located_points_new


def evaluation_sliSHAPs(file_ori, file, d, a, actual_points, interval, l):
    pkl_ori = pd.read_pickle(
        f'{root}/results/exp_2023_ijcai/{file_ori}_binned/full-1/ts_drifts_bi_predictions_Realworld_{file_ori}_binned_windowlength{d}_overlap{a}.pkl')
    slidSHAPs_ori = list(pkl_ori.values())[0]
    located_points_slidSHAPs_ori = getPosition(slidSHAPs_ori)

    pkl = pd.read_pickle(
        f'{root}/results/exp_2023_ijcai/{file}_binned/full-1/ts_drifts_bi_predictions_Realworld_{file}_binned_windowlength{d}_overlap{a}.pkl')
    slidSHAPs = list(pkl.values())[0]
    located_points_slidSHAPs = getPosition(slidSHAPs)
    located_points_slidSHAPs_new = reduce_FP(located_points_slidSHAPs_ori, located_points_slidSHAPs, actual_points, interval, l)
    total_num = len(slidSHAPs)

    print('=======================================================================================')
    print('slidSHAPs: ')
    print(f'total num: {total_num}')
    print(f'actual points: {actual_points}')
    print(f'located points slidSHAPs original: {located_points_slidSHAPs_ori}')
    print(f'located points slidSHAPs: {located_points_slidSHAPs}')
    print(f'located points slidSHAPs new: {located_points_slidSHAPs_new}')
    print('---------------------------------------------------------------------------------------')
    (drift_detection_dl_slidSHAPs, drift_detection_tp_slidSHAPs, drift_detection_fp_slidSHAPs,
     drift_detection_fn_slidSHAPs,
     accuracy_slidSHAPs, precision_slidSHAPs, recall_slidSHAPs, f1_slidSHAPs, delay_mean_slidSHAPs, delay_std_slidSHAPs) \
        = evaluator.DriftDetectionEvaluator.calculate_dl_tp_fp_fn(located_points_slidSHAPs_new, actual_points, interval,
                                                                  total_num)
    print(f'TP: {drift_detection_tp_slidSHAPs}, FP: {drift_detection_fp_slidSHAPs}, FN: {drift_detection_fn_slidSHAPs}')
    print(
        f'accuracy(A): {accuracy_slidSHAPs}, precision(P): {precision_slidSHAPs}, recall(R): {recall_slidSHAPs}, F1: {f1_slidSHAPs}')
    print(
        f'drift detection delay: {drift_detection_dl_slidSHAPs}, delay mean: {delay_mean_slidSHAPs}, delay std: {delay_std_slidSHAPs}')
    print('=======================================================================================')
    print()


def evaluation_HDDDM(file_ori, file, actual_points, interval, l):
    HDDDM_ori = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file_ori}_binned/full-1/HDDDM_bi_predictions_{file_ori}.csv',
                        delimiter=",", header=None).iloc[:, 0].tolist()
    located_points_HDDDM_ori = getPosition(HDDDM_ori)

    HDDDM = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_binned/full-1/HDDDM_bi_predictions_{file}.csv',
                        delimiter=",", header=None).iloc[:, 0].tolist()
    located_points_HDDDM = getPosition(HDDDM)
    located_points_HDDDM_new = reduce_FP(located_points_HDDDM_ori, located_points_HDDDM, actual_points, interval, l)
    total_num = len(HDDDM)

    print('=======================================================================================')
    print('HDDDM: ')
    print(f'total num: {total_num}')
    print(f'actual points: {actual_points}')
    print(f'located points HDDDM original: {located_points_HDDDM_ori}')
    print(f'located points HDDDM: {located_points_HDDDM}')
    print(f'located points HDDDM new: {located_points_HDDDM_new}')


    print('---------------------------------------------------------------------------------------')
    (drift_detection_dl_HDDDM, drift_detection_tp_HDDDM, drift_detection_fp_HDDDM,
     drift_detection_fn_HDDDM, accuracy_HDDDM, precision_HDDDM, recall_HDDDM, f1_HDDDM, delay_mean_HDDDM,
     delay_std_HDDDM) \
        = evaluator.DriftDetectionEvaluator.calculate_dl_tp_fp_fn(located_points_HDDDM_new, actual_points, interval,
                                                                  total_num)
    print(f'TP: {drift_detection_tp_HDDDM}, FP: {drift_detection_fp_HDDDM}, FN: {drift_detection_fn_HDDDM}')
    print(f'accuracy(A): {accuracy_HDDDM}, precision(P): {precision_HDDDM}, recall(R): {recall_HDDDM}, F1: {f1_HDDDM}')
    print(
        f'drift detection delay: {drift_detection_dl_HDDDM}, delay mean: {delay_mean_HDDDM}, delay std: {delay_std_HDDDM}')
    print('=======================================================================================')
    print()


def evaluation_ADWIN(file_ori, file, actual_points, interval, l):
    ADWIN_ori = pd.read_csv(
        f'{root}/results/exp_2023_ijcai/{file_ori}_binned/full-1/ADWIN_bi_predictions_{file_ori}.csv',
        delimiter=",", header=None).iloc[:, 0].tolist()
    located_points_ADWIN_ori = getPosition(ADWIN_ori)

    ADWIN = pd.read_csv(f'{root}/results/exp_2023_ijcai/{file}_binned/full-1/ADWIN_bi_predictions_{file}.csv',
                        delimiter=",", header=None).iloc[:, 0].tolist()
    located_points_ADWIN = getPosition(ADWIN)
    located_points_ADWIN_new = reduce_FP(located_points_ADWIN_ori, located_points_ADWIN, actual_points, interval, l)
    total_num = len(ADWIN)

    print('=======================================================================================')
    print('ADWIN: ')
    print(f'total num: {total_num}')
    print(f'actual points: {actual_points}')
    print(f'located points ADWIN original: {located_points_ADWIN_ori}')
    print(f'located points ADWIN: {located_points_ADWIN}')
    print(f'located points ADWIN new: {located_points_ADWIN_new}')

    print('---------------------------------------------------------------------------------------')
    (drift_detection_dl_ADWIN, drift_detection_tp_ADWIN, drift_detection_fp_ADWIN,
     drift_detection_fn_ADWIN, accuracy_ADWIN, precision_ADWIN, recall_ADWIN, f1_ADWIN, delay_mean_ADWIN,
     delay_std_ADWIN) \
        = evaluator.DriftDetectionEvaluator.calculate_dl_tp_fp_fn(located_points_ADWIN_new, actual_points, interval,
                                                                  total_num)
    print(f'TP: {drift_detection_tp_ADWIN}, FP: {drift_detection_fp_ADWIN}, FN: {drift_detection_fn_ADWIN}')
    print(f'accuracy(A): {accuracy_ADWIN}, precision(P): {precision_ADWIN}, recall(R): {recall_ADWIN}, F1: {f1_ADWIN}')
    print(
        f'drift detection delay: {drift_detection_dl_ADWIN}, delay mean: {delay_mean_ADWIN}, delay std: {delay_std_ADWIN}')
    print('=======================================================================================')
    print()


if __name__ == '__main__':

    # evaluation_sliSHAPs('BMW.DE_daily', 'BMW.DE(events)_daily', 30, 21, [1150, 2350, 3550, 4750, 5950], 300, 100)
    # evaluation_HDDDM('BMW.DE_daily', 'BMW.DE(events)_daily', [1150, 2350, 3550, 4750, 5950], 300)
    # evaluation_ADWIN('BMW.DE_daily', 'BMW.DE(events)_daily', [1150, 2350, 3550, 4750, 5950], 300)

    # evaluation_sliSHAPs('BMW.DE_hourly', 'BMW.DE(events2)_hourly', 30, 21, [950, 1950, 2950, 3950, 4950], 250, 100)
    # evaluation_HDDDM('BMW.DE_hourly', 'BMW.DE(events2)_hourly', [950, 1950, 2950, 3950, 4950], 250, 100)
    # evaluation_ADWIN('BMW.DE_hourly', 'BMW.DE(events2)_hourly', [950, 1950, 2950, 3950, 4950], 250, 100)

    evaluation_sliSHAPs('BMW.DE_minutely', 'BMW.DE(events2)_minutely', 30, 21, [275, 575, 875, 1175, 1475], 85, 50)
    evaluation_HDDDM('BMW.DE_minutely', 'BMW.DE(events2)_minutely', [275, 575, 875, 1175, 1475], 85, 50)
    evaluation_ADWIN('BMW.DE_minutely', 'BMW.DE(events2)_minutely', [275, 575, 875, 1175, 1475], 85, 50)

    # evaluation_sliSHAPs('Open_hourly', 'Open_hourly(events2)', 30, 21, [950, 1950, 2950, 3950, 4950], 250, 100)
    # evaluation_HDDDM('Open_hourly', 'Open_hourly(events)', [950, 1950, 2950, 3950, 4950], 250, 100)
    # evaluation_ADWIN('Open_hourly', 'Open_hourly(events)', [950, 1950, 2950, 3950, 4950], 250, 100)



    '''# accs(events2)_filled_binned
    located_points = [1329, 1989, 2529, 3459, 4179, 4299, 4809, 5649, 5979, 6429, 7419, 7659, 7899, 8049, 8349, 9399, 10119, 10539, 10839, 11559, 12399, 12999, 14259, 15429, 15999, 16599, 17229, 17859, 18429]
    actual_points = [3000, 6000, 9000, 12000, 15000]  # 实际的漂移点位置列表
    interval = 750  # 漂移检测的时间间隔
    total_num = 18479  # 总的数据点数量'''
