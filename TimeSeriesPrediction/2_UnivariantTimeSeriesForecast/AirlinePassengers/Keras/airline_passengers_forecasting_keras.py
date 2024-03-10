import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Normalizer, LSTM, LSTM-Architecture, Beschreibung


class AirlinePassengersDataSet:
    def __init__(self):
        self.data = pd.read_csv("../AirlinePassengers.csv", sep=";")
        self.threshold = 107

    def get_train_data(self):
        data = self.data.loc[0:self.threshold, "Passengers"].reset_index(drop=True)
        return pd.DataFrame(data)

    def get_test_data(self):
        data = self.data.loc[self.threshold:142, "Passengers"].reset_index(drop=True)
        return pd.DataFrame(data)


airlinePassengers = AirlinePassengersDataSet()
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(airlinePassengers.get_train_data())
test = scaler.fit_transform(airlinePassengers.get_test_data())


def create_timeseries(data, previous=1):
    inputs, targets = list(), list()
    for i in range(len(data)-previous-1):
        a = data[i: (i+previous), 0]
        inputs.append(a)
        targets.append(data[i + previous, 0])
    return np.array(inputs), np.array(targets)


lookback = 15
train_inputs, train_targets = create_timeseries(train, lookback)
test_inputs, test_targets = create_timeseries(test, lookback)

train_inputs = np.reshape(train_inputs, (train_inputs.shape[0], 1, train_inputs.shape[1]))
test_inputs = np.reshape(test_inputs, (test_inputs.shape[0], 1, test_inputs.shape[1]))


def create_FFN(inputs, targets):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=8, activation="relu"))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')
    model.fit(inputs, targets, epochs=25, batch_size=1)
    return model


ffn = create_FFN(train_inputs, train_targets)


def validation_forecast(model, inputs):
    predictions = model.predict(inputs)
    return predictions.flatten()


validation = pd.DataFrame()
validation["true"] = scaler.inverse_transform([test_targets]).flatten()
validation["validation"] = scaler.inverse_transform([validation_forecast(ffn, test_inputs)]).flatten()
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


start_value = train_inputs[-1]
number_of_predictions = 40

prediction = pd.DataFrame()
prediction["one_step_prediction"] = scaler.inverse_transform([one_step_ahead_forecast(ffn, start_value, number_of_predictions)]).flatten()
prediction.index += airlinePassengers.threshold

plt.plot(airlinePassengers.get_train_data(), color="green", label="training")
plt.plot(validation["true"], color="red", label="true")
plt.plot(validation["validation"], color="blue", label="validation")
plt.plot(prediction["one_step_prediction"], color="orange", label="one_step_prediction")
plt.title("airline passengers prediction")
plt.xlabel("Time[Month]")
plt.ylabel("Passengers[x1000]")
plt.legend(loc="upper left")
plt.savefig("./airlinePassengers_keras.png")
plt.show()
