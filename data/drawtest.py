import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    data = np.asarray(pd.read_csv(f'Synthetic_6Concepts_MUL_5000dataPerConcept_4category.txt', delimiter=",", header=None))

    plt.plot(data, linewidth=0.1)
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.title('Plot of the Data')
    plt.show()
