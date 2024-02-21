import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class AirlinePassengersDataSet:
    def __init__(self):
        self.data = pd.read_csv("./AirlinePassengers.csv", sep=";")
        self.data.drop(["Month"], inplace=True, axis=1)

    def create_targets(self):
        passengers_shift_1 = self.data.loc[:, "Passengers"].shift(-1)
        self.data["Passengers+1"] = passengers_shift_1
        self.data = self.data[:-2]

    def get_train_test(self):
        train = self.data[0:107]
        test = self.data[107:142]
        return train, test


timeseries = AirlinePassengersDataSet()
timeseries.create_targets()
train, test = timeseries.get_train_test()

# LSTM input shape = (samples, timesteps, features)
train_inputs_array = train.loc[:, "Passengers"].to_numpy()
train_inputs_array = train_inputs_array.reshape(train_inputs_array.shape[0], 1, 1)
train_targets = train.loc[:, "Passengers+1"]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=8, activation="relu"))
model.add(tf.keras.layers.Dense(units=1))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')
model.fit(train_inputs_array, train_targets, epochs=25, batch_size=1)

results = pd.DataFrame()
results["true"] = test["Passengers"]


def validation_forecast(inputs):
    predictions = model.predict(inputs)
    return predictions.flatten().tolist()


validation_inputs = test.loc[:, "Passengers"]
results["validation"] = validation_forecast(validation_inputs)


def one_step_ahead_forecast(current_value):
    one_step_ahead_forecast = list()
    for element in range(0, len(test)):
        prediction = model.predict([current_value])
        one_step_ahead_forecast.append(prediction[0][0][0])
        current_value = prediction[0]
    return one_step_ahead_forecast


start_value = train.iloc[-1:, 1]
results["one_step_prediction"] = one_step_ahead_forecast(start_value)

plt.plot(train["Passengers"], color="green", label="training")
plt.plot(results["true"], color="red", label="prediction")
plt.plot(results["validation"], color="blue", label="test")
plt.plot(results["one_step_prediction"], color="orange", label="one_step_prediction")
plt.title("airline passengers prediction")
plt.xlabel("Time[Month]")
plt.ylabel("Passengers[x1000]")
plt.legend(loc="upper left")
plt.savefig("./img/airlinePassengers_keras.png")
plt.show()