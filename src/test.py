# Imports
import numpy as np
import pandas as pd
from skmultiflow.drift_detection import DDM
from pathlib import Path

if __name__ == '__main__':
    ddm = DDM()
    # Simulating a data stream as a normal distribution of 1's and 0's
    root = Path(__file__).resolve().parent.parent
    data_stream = np.asarray(
        pd.read_csv(f'{root}/data/' + 'Synthetic_6Concepts_ADD_5000dataPerConcept_10category.csv', delimiter=",",
                    header=None))
    data_stream = np.transpose(data_stream)
    print(data_stream.shape)

    # Assuming each row of data_stream is a data point and you are making predictions
    for i in range(data_stream.shape[0]):
        for j in range(data_stream.shape[1]):
            prediction = your_model.predict(data_stream[i, j])  # Replace 'your_model' with your actual model
            ddm.add_element(prediction)
            if ddm.detected_warning_zone():
                print(f'Warning zone has been detected in prediction: {prediction} - at index: ({i}, {j})')
            if ddm.detected_change():
                print(f'Change has been detected in prediction: {prediction} - at index: ({i}, {j})')

