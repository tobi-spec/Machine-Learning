import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class AirlinePassengersDataSet:
    def __init__(self):
        self.data = pd.read_csv("../../AirlinePassengers.csv", sep=";")
        self.threshold = 107

    def get_train_data(self):
        data = self.data.loc[0:self.threshold, "Passengers"].reset_index(drop=True)
        return pd.DataFrame(data)

    def get_test_data(self):
        data = self.data.loc[self.threshold:142, "Passengers"].reset_index(drop=True)
        return pd.DataFrame(data)


airlinePassengers = AirlinePassengersDataSet()
train = airlinePassengers.get_train_data()
test = airlinePassengers.get_test_data()


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
        return self.data.loc[element]

    def __get_timeseries(self, element):
        return self.data.loc[element-self.lookback: element-1].to_list()


lookback = 30
train_inputs, train_targets = TimeSeriesGenerator(train, lookback).create_timeseries()
test_inputs, test_targets = TimeSeriesGenerator(test, lookback).create_timeseries()


def create_LSTM_model(inputs, targets):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=50,
                                    activation="relu",
                                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                    bias_initializer=tf.keras.initializers.Zeros())
    )
    model.add(tf.keras.layers.Dense(units=1,
                                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                    bias_initializer=tf.keras.initializers.Zeros()))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='mean_squared_error')
    model.fit(inputs, targets, epochs=100, batch_size=1)
    return model


model = create_LSTM_model(train_inputs, train_targets)


def validation_forecast(model, inputs):
    predictions = model.predict(inputs)
    return predictions.flatten()


validation_results = validation_forecast(model, test_inputs)

validation = pd.DataFrame()
validation["true"] = test_targets
validation["validation"] = validation_results
validation.index += airlinePassengers.threshold


def one_step_ahead_forecast(model, current_value, number_of_predictions):
    one_step_ahead_forecast = list()
    for element in range(0, number_of_predictions):
        prediction = model.predict([current_value])
        one_step_ahead_forecast.append(prediction[0][0])
        current_value = np.delete(current_value, 0)
        current_value = np.append(current_value, prediction)
        current_value = current_value.reshape(1, current_value.shape[0])
    return one_step_ahead_forecast


start_value = test_inputs[0]
start_value_reshaped = start_value.reshape(1, start_value.shape[0])
number_of_predictions = 40
prediction_results = one_step_ahead_forecast(model, start_value_reshaped, number_of_predictions)

prediction = pd.DataFrame()
prediction["one_step_prediction"] = prediction_results
prediction.index += airlinePassengers.threshold

plt.plot(airlinePassengers.data["Passengers"], color="pink", label="dataset")
plt.plot(airlinePassengers.get_train_data(), color="green", label="training")
plt.plot(validation["true"], color="red", label="true")
plt.plot(validation["validation"], color="blue", label="validation")
plt.plot(prediction["one_step_prediction"], color="orange", label="one_step_prediction")
plt.title("airline passengers prediction FF")
plt.xlabel("Time[Month]")
plt.ylabel("Passengers[x1000]")
plt.xticks(range(0, 150, 20))
plt.yticks(range(0, 1000, 100))
plt.legend(loc="upper left")
plt.savefig("./airlinePassengers_keras_ff.png")
plt.show()
