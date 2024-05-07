import pandas as pd
import numpy as np
from neuronal_network_types import NeuronalNetworkTypes


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


class Forecaster:
    def __init__(self, model, start_value, number_of_predictions, output_dimension_type):
        self.model = model
        self.current_value = start_value
        self.number_of_predictions = number_of_predictions
        self.output_dimension_type = output_dimension_type

    def one_step_ahead(self):
        one_step_ahead_forecast = list()
        for element in range(0, self.number_of_predictions):
            prediction = self.model.predict(self.current_value)
            one_step_ahead_forecast.append(prediction[0][0])
            self.current_value = self.__move_numpy_queue(prediction)
        return one_step_ahead_forecast

    def __move_numpy_queue(self, prediction):
        self.current_value = np.delete(self.current_value, 0)
        self.current_value = np.append(self.current_value, prediction)
        return self.__format_dimension()

    def __format_dimension(self):
        match self.output_dimension_type:
            case NeuronalNetworkTypes.LSTM:
                return self.current_value.reshape(1, 1, self.current_value.shape[0])
            case NeuronalNetworkTypes.ATTENTION | NeuronalNetworkTypes.CNN:
                return self.current_value.reshape(1, self.current_value.shape[0], 1)
            case NeuronalNetworkTypes.FEED_FORWARD:
                return self.current_value.reshape(1, self.current_value.shape[0])
            case _:
                return self.current_value
