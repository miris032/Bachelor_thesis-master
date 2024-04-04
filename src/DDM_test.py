import numpy as np
from matplotlib import pyplot as plt
from skmultiflow.drift_detection.eddm import EDDM
from pathlib import Path

root = Path(__file__).resolve().parent.parent

if __name__ == '__main__':
    # Imports
    import numpy as np
    from skmultiflow.drift_detection.hddm_w import HDDM_W

    hddm_w = HDDM_W()
    # Simulating a data stream as a normal distribution of 1's and 0's
    data_stream = np.random.uniform(1.5, 2, size=200)
    # Changing the data concept from index 999 to 1500, simulating an
    # increase in error rate
    for i in range(99, 150):
        data_stream[i] = 1
    # Adding stream elements to HDDM_A and verifying if drift occurred
    for i in range(200):
        hddm_w.add_element(data_stream[i])
        '''if hddm_w.detected_warning_zone():
            print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))'''
        if hddm_w.detected_change():
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))

    plt.plot(data_stream, linewidth=0.3, alpha=0.5)
    plt.show()

    '''# Visualisation of the data stream
    plt.figure(figsize=(10, 5))
    for column in data_stream.columns:
        plt.plot(data_stream[column], label=column, linewidth=0.3, alpha=0.5)
    plt.title('Generated Data Stream')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()'''
