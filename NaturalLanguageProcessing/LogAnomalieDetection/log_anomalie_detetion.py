import pandas as pd
import re


class LogParser:
    def __init__(self):
        self.logs = self._load_logs("./log_example.txt")

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


if __name__ == "__main__":
    log_parser = LogParser()
    print(log_parser.get_all_data())

