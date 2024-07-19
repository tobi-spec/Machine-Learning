from os import listdir
from TimeSeriesPrediction.AnomalieDetection.LogFiles.log_anomalie_detection_keras_utils import LogParser
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


def create_logfile():
    logs = LogParser("data/correct/log_original.txt").get_all_data()
    number_of_file = len(listdir("./data/correct"))
    number_of_changes = np.random.randint(low=3, high=13)
    positions_of_changes = np.random.randint(low=1, high=49, size=number_of_changes)
    for position in positions_of_changes:
        swap1 = logs.iloc[position, 1:4]
        swap2 = logs.iloc[position+1, 1:4]

        temp = swap1.copy()
        logs.iloc[position, 1:4] = swap2
        logs.iloc[position+1, 1:4] = temp

    logs.to_csv(f'./data/correct/log{number_of_file}.txt', index=False, header=False)


for i in range(0, 20):
    create_logfile()




