import numpy as np
import pandas as pd
import re
import os


class LogParser:
    def __init__(self, path: str):
        self.logs = self._load_logs(path)

    def _load_logs(self, file_path):
        with open(file_path, 'r') as file:
            log_lines = file.readlines()

        log_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (Component\d+) (\w+) (.*)')
        log_data = []
        for line in log_lines:
            match = log_pattern.match(line.strip())
            if match:
                log_data.append(match.groups())
        return pd.DataFrame(log_data, columns=['Timestamp', 'Component', 'Level', 'Message'])

    def get_all_data(self):
        return self.logs

    def get_timestamps(self):
        return self.logs['Timestamp']

    def get_components(self):
        return self.logs['Component']

    def get_levels(self):
        return self.logs['Level']

    def get_messages(self):
        return self.logs['Message']

    def get_numbers(self):
        # try/catch block
        return self.logs["Number"]


class DataBuilder(LogParser):
    def __init__(self, path: str, translation_table: dict):
        super().__init__(path)
        self.translation_table = translation_table
        self.data = None

    def add_number_representation(self):
        numbers = list()
        for message in self.logs['Message']:
            numbers.append(self.translation_table[message])
        self.logs["Number"] = numbers
        return self


def transform_messages_to_numbers(logs):
    return dict({entry: index for index, entry in enumerate(logs)})


def create_train_data(translation_table):
    log_files = os.listdir("../../data/correct")
    list_of_timeseries = list()
    for log in log_files:
        timeseries = DataBuilder(f"../../data/correct/{log}",
                                 translation_table).add_number_representation().get_numbers()
        list_of_timeseries.append(timeseries)
    return np.array(list_of_timeseries)
