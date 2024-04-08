import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class BeijingDataSet:
    def __init__(self):
        self.dataset = pd.read_csv(
            filepath_or_buffer="BeijingParticulateMatter.csv",
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
        self.dataset.to_csv("./beijing_pollution.csv")

    def get_train(self):
        data = self.dataset[:self.threshold]
        return pd.DataFrame(data)

    def get_test(self):
        data = self.dataset[self.threshold:]
        return pd.DataFrame(data)


beijingData = BeijingDataSet()
beijingData.encode_labels()
train = beijingData.get_train()
test = beijingData.get_test()


scaler = MinMaxScaler((0, 1))
scaled_train = scaler.fit_transform(train)
scaled_test = scaler.fit_transform(test)


class TimeSeriesGenerator:
    def __init__(self, data, lookback):
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


lookback = 30
train_timeseries, train_targets = TimeSeriesGenerator(scaled_train, lookback).create_timeseries()
test_timeseries, test_targets = TimeSeriesGenerator(scaled_test, lookback).create_timeseries()


def create_FF_model(inputs, targets):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=50,
                                    activation="relu",
                                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                    bias_initializer=tf.keras.initializers.Zeros())
    )
    model.add(tf.keras.layers.Dense(units=50,
                                    activation="relu",
                                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                    bias_initializer=tf.keras.initializers.Zeros())
    )
    model.add(tf.keras.layers.Dense(units=8,
                                    activation="relu",
                                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                    bias_initializer=tf.keras.initializers.Zeros()))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='mean_squared_error')
    model.fit(inputs, targets, epochs=2, batch_size=16)
    return model


model = create_FF_model(train_timeseries, train_targets)


def one_step_ahead_forecast(model, current_value, number_of_predictions):
    one_step_ahead_forecast = list()
    for element in range(0, number_of_predictions):
        prediction = model.predict([current_value])
        one_step_ahead_forecast.append(prediction[0][0])
        current_value = np.delete(current_value, 0, axis=1)
        temp = prediction[0][0]
        temp = temp.reshape(1, 1, temp.shape[0])
        current_value = np.concatenate((current_value, temp), axis=1)
    return np.array(one_step_ahead_forecast)


start_index = -1
start_value = train_timeseries[start_index]
start_value_reshaped = start_value.reshape(1, start_value.shape[0], start_value.shape[1])
number_of_predictions = 80
prediction_results = one_step_ahead_forecast(model, start_value_reshaped, number_of_predictions)

prediction_rescaled = scaler.inverse_transform(prediction_results)

prediction = pd.DataFrame(prediction_rescaled)
prediction.columns = beijingData.dataset.columns


def create_dates_for_forecast(start, number):
    indicies = list()
    indicies.append(start)
    for element in range(1, number):
        indicies.append(indicies[-1] + pd.Timedelta(hours=1))
    return indicies


prediction.index = create_dates_for_forecast(train.index[-1], number_of_predictions)

plt.plot(beijingData.dataset["pollution"], color="red", label="dataset")
plt.plot(beijingData.get_train()["pollution"], color="green", label="training")
plt.plot(prediction["pollution"], color="orange", label="one_step_prediction")
plt.title("beijing pollution prediction FF")
plt.xlabel("Time[Month]")
plt.ylabel("Passengers[x1000]")
plt.legend(loc="upper left")
plt.savefig("./beijing_keras_ff.png")
plt.show()

# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
