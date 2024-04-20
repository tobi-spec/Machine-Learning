import pandas as pd
import numpy as np

class AirlinePassengersDataSet:
    def __init__(self):
        self.data = pd.read_csv("../../AirlinePassengers.csv", sep=";")
        self.threshold = 107

    def get_train_data(self):
        data = self.data.loc[0:self.threshold, "Passengers"].reset_index(drop=True)
        return data.to_numpy()

    def get_test_data(self):
        data = self.data.loc[self.threshold:142, "Passengers"].reset_index(drop=True)
        return data.to_numpy()


class TimeSeriesGenerator:
    def __init__(self, data, lookback):
        self.data = data
        self.lookback = lookback

    def create_timeseries(self):
        inputs, targets = list(), list()
        for element in range(self.lookback, len(self.data)-1):
            inputs.append(self.__get_timeseries(element))
            targets.append(self.__get_targets(element))
        return np.array(inputs), np.array(targets)

    def __get_targets(self, element):
        return self.data[element]

    def __get_timeseries(self, element):
        return self.data[element-self.lookback: element]


def validation_forecast(model, inputs):
    predictions = model.predict(inputs)
    return predictions.flatten()
