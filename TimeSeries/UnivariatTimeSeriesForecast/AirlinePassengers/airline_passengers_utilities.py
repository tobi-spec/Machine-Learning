import pandas as pd
import numpy as np
from pathlib import Path
from neuronal_network_types import NeuronalNetworkTypes
import matplotlib.pyplot as plt


class AirlinePassengersDataSet:
    def __init__(self):
        path = Path(__file__) / "../AirlinePassengers.csv"
        self.data = pd.read_csv(path, sep=";")
        self.threshold = 107

    @property
    def passengers(self) -> np.ndarray:
        data = self.data["Passengers"]
        return data.to_numpy()

    @property
    def train_data(self) -> np.ndarray:
        data = self.data.loc[0:self.threshold, "Passengers"].reset_index(drop=True)
        return data.to_numpy()

    @property
    def test_data(self) -> np.ndarray:
        data = self.data.loc[self.threshold:142, "Passengers"].reset_index(drop=True)
        return data.to_numpy()


class TimeSeriesGenerator:
    def __init__(self, data: np.ndarray, lookback: int, lookout: int):
        self.data = data
        self.lookback = lookback
        self.lookout = lookout

    def create_timeseries(self) -> (np.ndarray, np.ndarray):
        inputs, targets = list(), list()
        for element in range(self.lookback, len(self.data)-self.lookout-1):
            inputs.append(self._get_timeseries(element))
            targets.append(self._get_targets(element))
        return np.array(inputs), np.array(targets)

    def _get_targets(self, element) -> np.ndarray:
        return self.data[element: element+self.lookout]

    def _get_timeseries(self, element) -> np.ndarray:
        return self.data[element-self.lookback: element]


def keras_forecast(model, inputs):
    predictions = model.predict(inputs)
    return predictions.flatten()


class KerasForecaster:
    def __init__(self, model, start_value, number_of_predictions, output_dimension_type):
        self.model = model
        self.current_value = start_value
        self.number_of_predictions = number_of_predictions
        self.output_dimension_type = output_dimension_type

    def one_step_ahead(self):
        one_step_ahead_forecast = list()
        for element in range(0, self.number_of_predictions):
            prediction = self.model.predict(self.current_value)
            one_step_ahead_forecast.append(self._getFeature(prediction))
            self.current_value = self._move_numpy_queue(prediction)
        return one_step_ahead_forecast

    def _getFeature(self, prediction):
        return prediction[0][0]

    def _move_numpy_queue(self, prediction):
        self.current_value = np.delete(self.current_value, 0)
        self.current_value = np.append(self.current_value, prediction)
        return self.__format_dimension()

    def __format_dimension(self):
        match self.output_dimension_type:
            case NeuronalNetworkTypes.ATTENTION | NeuronalNetworkTypes.CNN:
                return shape_batch_timestamp_feature(self.current_value)
            case NeuronalNetworkTypes.LSTM | NeuronalNetworkTypes.RNN:
                # recurrent networks seems to work better with switch timestamp/feature
                return shape_batch_feature_timestamp(self.current_value)
            case NeuronalNetworkTypes.FEED_FORWARD:
                return shape_batch_timestamp(self.current_value)
            case _:
                return self.current_value


class Seq2SeqKerasForecaster(KerasForecaster):
    def __init__(self, model, start_value, number_of_predictions, output_dimension_type, current_target):
        KerasForecaster.__init__(self, model, start_value, number_of_predictions, output_dimension_type)
        self.current_target = current_target

    def one_step_ahead(self):
        forecast = list()
        for element in range(0, self.number_of_predictions):
            prediction = self.model.predict([self.current_value, self.current_target])
            forecast.append(self.__getFeature(prediction))
            self.current_value = KerasForecaster._move_numpy_queue(self, prediction)
        return forecast

    def __getFeature(self, prediction):
        return prediction[0][0][0]


def shape_batch_feature_timestamp(value):
    return value.reshape(1, 1, value.shape[0])


def shape_batch_timestamp_feature(value):
    return value.reshape(1, value.shape[0], 1)


def shape_batch_timestamp(value):
    return value.reshape(1, value.shape[0])


def plot_results(prediction: pd.Series, validation: pd.Series, name):
    airline_passengers = AirlinePassengersDataSet()
    plt.plot(airline_passengers.passengers, color="red", label="dataset")
    plt.plot(airline_passengers.train_data, color="green", label="training")
    plt.plot(validation, color="blue", label="validation")
    plt.plot(prediction, color="orange", label="one_step_prediction")
    plt.title(f"airline passengers {name}")
    plt.xlabel("Time[Month]")
    plt.ylabel("Passengers[x1000]")
    plt.xticks(range(0, 200, 20))
    plt.yticks(range(0, 1000, 100))
    plt.legend(loc="upper left")
    plt.savefig(f"./airline_passengers_keras_{name}.png")
    plt.show()
