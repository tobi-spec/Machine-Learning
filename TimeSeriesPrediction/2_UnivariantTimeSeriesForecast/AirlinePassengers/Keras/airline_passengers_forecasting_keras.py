import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Normalizer, LSTM, LSTM-Architecture, Beschreibung


class AirlinePassengersDataSet:
    def __init__(self):
        self.data = pd.read_csv("../AirlinePassengers.csv", sep=";")
        self.data.drop(["Month"], inplace=True, axis=1)
        self.threshold = 107

    def create_targets(self):
        self.data["Passengers+1"] = self.data.loc[:, "Passengers"].shift(-1)
        self.data = self.data[:-2]

    def get_train_inputs(self):
        return self.data.loc[0:self.threshold, "Passengers"].reset_index(drop=True)

    def get_train_targets(self):
        return self.data.loc[0:self.threshold, "Passengers+1"].reset_index(drop=True)

    def get_test_inputs(self):
        return self.data.loc[self.threshold:142, "Passengers"].reset_index(drop=True)

    def get_test_targets(self):
        return self.data.loc[self.threshold:142, "Passengers+1"].reset_index(drop=True)


airlinePassengers = AirlinePassengersDataSet()
airlinePassengers.create_targets()


def create_timeseries(inputs, targets, span):
    timeseries = list()
    for i in range(span, len(inputs)):
        timeseries.append(inputs.loc[i - span:i])
    timeseries = np.array(timeseries)
    targets = targets[span:]
    return timeseries, targets

size_of_timeseries = 15
train_input_timeseries, train_targets = create_timeseries(airlinePassengers.get_train_inputs(),
                                                          airlinePassengers.get_train_targets(), size_of_timeseries)

test_input_timeseries, test_targets = create_timeseries(airlinePassengers.get_test_inputs(),
                                                        airlinePassengers.get_test_targets(), size_of_timeseries)


def create_FFN(inputs, targets):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=8, activation="relu"))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')
    model.fit(inputs, targets, epochs=25, batch_size=1)
    return model


ffn = create_FFN(train_input_timeseries, train_targets)


def validation_forecast(model, inputs):
    predictions = model.predict(inputs)
    return predictions.flatten()


validation = pd.DataFrame()
validation["true"] = test_targets.shift(1)
validation["validation"] = validation_forecast(ffn, test_input_timeseries)
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


start_value = train_input_timeseries[-1]
start_value_reshaped = start_value.reshape(1, start_value.shape[0])
number_of_predictions = 40

prediction = pd.DataFrame()
prediction["one_step_prediction"] = one_step_ahead_forecast(ffn, start_value_reshaped, number_of_predictions)
prediction.index += airlinePassengers.threshold

plt.plot(airlinePassengers.get_train_inputs(), color="green", label="training")
plt.plot(validation["true"], color="red", label="true")
plt.plot(validation["validation"], color="blue", label="validation")
plt.plot(prediction["one_step_prediction"], color="orange", label="one_step_prediction")
plt.title("airline passengers prediction")
plt.xlabel("Time[Month]")
plt.ylabel("Passengers[x1000]")
plt.legend(loc="upper left")
plt.savefig("./airlinePassengers_keras.png")
plt.show()
