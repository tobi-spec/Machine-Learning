import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from TimeSeriesPrediction.UnivariatTimeSeriesForecast.AirlinePassengers.airline_passengers_utilities import *


def main():
    airline_passengers = AirlinePassengersDataSet()
    train_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = train_scaler.fit_transform(airline_passengers.get_train_data().reshape(-1, 1))

    test_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_test = test_scaler.fit_transform(airline_passengers.get_test_data().reshape(-1, 1))

    lookback: int = 30
    train_timeseries, train_targets = TimeSeriesGenerator(scaled_train, lookback).create_timeseries()
    test_timeseries, test_targets = TimeSeriesGenerator(scaled_test, lookback).create_timeseries()

    model = LSTMModel(lookback)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')
    model.fit(train_timeseries, train_targets, epochs=1000, batch_size=1)

    validation_results = validation_forecast(model, test_timeseries)

    validation = pd.DataFrame()
    validation["validation"] = test_scaler.inverse_transform([validation_results]).flatten()
    validation.index += airline_passengers.threshold + lookback

    start_index = -1
    start_value = train_timeseries[start_index]
    start_value_reshaped = start_value.reshape(1, start_value.shape[0], start_value.shape[1])
    number_of_predictions = 80
    prediction_results = one_step_ahead_forecast(model, start_value_reshaped, number_of_predictions)

    prediction = pd.DataFrame()
    prediction["one_step_prediction"] = train_scaler.inverse_transform([prediction_results]).flatten()
    prediction.index += airline_passengers.threshold + start_index

    plt.plot(airline_passengers.data["Passengers"], color="red", label="dataset")
    plt.plot(airline_passengers.get_train_data(), color="green", label="training")
    plt.plot(validation["validation"], color="blue", label="validation")
    plt.plot(prediction["one_step_prediction"], color="orange", label="one_step_prediction")
    plt.title("airline passengers prediction LSTM")
    plt.xlabel("Time[Month]")
    plt.ylabel("Passengers[x1000]")
    plt.xticks(range(0, 200, 20))
    plt.yticks(range(0, 1000, 100))
    plt.legend(loc="upper left")
    plt.savefig("./img/airlinePassengers_keras_lstm.png")
    plt.show()


# training scedular learning rate wird angepasst nach x epochs
# Bidirectionales lernen - Zeitreihe umkehren - https://keras.io/examples/nlp/bidirectional_lstm_imdb/
# Kompletten daten fürs Training nehmen
# Masked traning - Lücken in Traningsdaten schließen
class LSTMModel(tf.keras.Model):
    def __init__(self, lookback):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(units=50,
                                   activation="tanh",
                                   recurrent_activation="sigmoid",
                                   input_shape=(lookback, 1),
                                   kernel_initializer="glorot_uniform",
                                   recurrent_initializer="orthogonal",
                                   bias_initializer="zeros",
                                   )
        self.dense1 = tf.keras.layers.Dense(units=1,
                                            activation="relu",
                                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                            bias_initializer=tf.keras.initializers.Zeros())

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense1(x)
        return x


def one_step_ahead_forecast(model, current_value, number_of_predictions):
    one_step_ahead_forecast = list()
    for element in range(0, number_of_predictions):
        prediction = model.predict(current_value)
        one_step_ahead_forecast.append(prediction[0][0])
        current_value = np.delete(current_value, 0)
        current_value = np.append(current_value, prediction)
        current_value = current_value.reshape(1, current_value.shape[0], 1)
    return one_step_ahead_forecast


if __name__ == "__main__":
    main()