from os import listdir
from TimeSeriesPrediction.AnomalieDetection.LogFiles.log_anomalie_detection_keras_utils import LogParser
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


def create_logfile(logfile):
    number_of_file = len(listdir("./data/correct"))
    number_of_changes = np.random.randint(low=3, high=13)
    positions_of_changes = np.random.randint(low=1, high=49, size=number_of_changes)
    for position in positions_of_changes:
        swap1 = logfile.iloc[position, 1:4]
        swap2 = logfile.iloc[position+1, 1:4]

        temp = swap1.copy()
        logfile.iloc[position, 1:4] = swap2
        logfile.iloc[position+1, 1:4] = temp

    logfile.to_csv(f'./data/correct/log{number_of_file}.txt', index=False, header=False)


logs = LogParser("data/correct/log_original.txt").get_all_data()
for i in range(0, 10):
    create_logfile(logs)




