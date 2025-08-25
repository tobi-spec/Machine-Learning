import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras import layers, optimizers, initializers, Sequential
import matplotlib.pyplot as plt


def main():
    beijing_data = BeijingDataSet()
    beijing_data.encode_labels()
    train: pd.DataFrame = beijing_data.train
    test: pd.DataFrame = beijing_data.test

    scaler = MinMaxScaler((0, 1))
    scaled_train: np.array = scaler.fit_transform(train)
    scaled_test: np.array = scaler.fit_transform(test)

    lookback: int = 30
    train_timeseries, train_targets = TimeSeriesGenerator(scaled_train, lookback).create_timeseries()
    test_timeseries, test_targets = TimeSeriesGenerator(scaled_test, lookback).create_timeseries()

    feed_forward_model = create_FF_model(train_timeseries, train_targets)

    start_index: int = -1
    start_value: np.array = train_timeseries[start_index]
    start_value_reshaped: np.array = start_value.reshape(1, start_value.shape[0], start_value.shape[1])
    number_of_predictions: int = 80
    prediction_results: np.array = Forecaster(feed_forward_model, start_value_reshaped,
                                              number_of_predictions).one_step_ahead_forecast()

    prediction_rescaled: np.array = scaler.inverse_transform(prediction_results)

    prediction = pd.DataFrame(prediction_rescaled)
    prediction.columns = beijing_data.dataset.columns

    prediction.index = create_dates_for_forecast(train.index[-1], number_of_predictions)

    plt.plot(beijing_data.dataset["pollution"], color="red", label="dataset")
    plt.plot(beijing_data.train["pollution"], color="green", label="training")
    plt.plot(prediction["pollution"], color="orange", label="one_step_prediction")
    plt.title("beijing pollution prediction FF")
    plt.xlabel("Time[Month]")
    plt.ylabel("Passengers[x1000]")
    plt.legend(loc="upper left")
    plt.savefig("./beijing_keras_ff.png")
    plt.show()

class BeijingDataSet:
    def __init__(self):
        self.dataset: pd.DataFrame = pd.read_csv(
            filepath_or_buffer="../BeijingParticulateMatter.csv",
            delimiter=",",
            index_col=0,
            parse_dates=[[1, 2, 3, 4]],
            date_format='%Y %m %d %H')

        self.dataset.drop(["No"], axis=1, inplace=True)
        self.dataset.columns = ['pollution', 'dew', 'temperature', 'pressure', 'wind_direction', 'wind_speed', 'snow', 'rain']
        self.dataset.index.name = 'date'
        self.dataset.loc[:, "pollution"].fillna(0, inplace=True)
        self.dataset = self.dataset[24:] #First 24 entries have NaN in 'pollution' column
        self.threshold = int(len(self.dataset)*0.8)

    def encode_labels(self):
        encoder = LabelEncoder()
        self.dataset.loc[:, "wind_direction"] = encoder.fit_transform(self.dataset.loc[:, "wind_direction"])

    def save(self):
        self.dataset.to_csv("../beijing_pollution.csv")

    @property
    def train(self):
        data = self.dataset[:self.threshold]
        return pd.DataFrame(data)

    @property
    def test(self):
        data = self.dataset[self.threshold:]
        return pd.DataFrame(data)


class TimeSeriesGenerator:
    def __init__(self, data: np.array, lookback: int):
        self.data = data
        self.lookback = lookback

    def create_timeseries(self):
        inputs, targets = list(), list()
        for element in range(self.lookback, len(self.data)-1):
            inputs.append(self.__get_timeseries(element))
            targets.append([self.__get_targets(element)])
        return np.array(inputs), np.array(targets)

    def __get_targets(self, element):
        return self.data[element]

    def __get_timeseries(self, element):
        return self.data[element-self.lookback: element]


def create_FF_model(inputs: np.array, targets: np.array) -> Sequential:
    model = Sequential()
    model.add(layers.Dense(units=50,
                                    activation="relu",
                                    kernel_initializer=initializers.RandomNormal(stddev=0.01),
                                    bias_initializer=initializers.Zeros())
    )
    model.add(layers.Dense(units=50,
                                    activation="relu",
                                    kernel_initializer=initializers.RandomNormal(stddev=0.01),
                                    bias_initializer=initializers.Zeros())
    )
    model.add(layers.Dense(units=8,
                                    activation="relu",
                                    kernel_initializer=initializers.RandomNormal(stddev=0.01),
                                    bias_initializer=initializers.Zeros()))
    model.compile(optimizer=optimizers.Adam(0.0001), loss='mean_squared_error')
    model.fit(inputs, targets, epochs=2, batch_size=16)
    return model


class Forecaster:
    def __init__(self, model: Sequential, start_value: np.array, number_of_forecasts: int):
        self.model = model
        self.current_value = start_value
        self.forecasts = number_of_forecasts

    def one_step_ahead_forecast(self):
        forecast = list()
        for element in range(0, self.forecasts):
            predictions = self.model.predict([self.current_value])
            features = self.__getFeatures(predictions)
            forecast.append(features)
            self.current_value = self.__move_numpy_queue(features)
        return np.array(forecast)

    def __move_numpy_queue(self, features):
        current_value = np.delete(self.current_value, 0, axis=1)
        temp = features.reshape(1, 1, features.shape[0])
        return np.concatenate((current_value, temp), axis=1)

    def __getFeatures(self, prediction):
        return prediction[0][0]


def create_dates_for_forecast(start: pd.Timestamp, number: int):
    indices = list()
    indices.append(start)
    for element in range(1, number):
        indices.append(indices[-1] + pd.Timedelta(hours=1))
    return indices


if __name__ == "__main__":
    main()
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
