import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class AirlinePassengersDataSet:
    def __init__(self):
        self.data = pd.read_csv("./AirlinePassengers.csv", sep=";")
        self.data.drop(["Month"], inplace=True, axis=1)

    def create_targets(self):
        self.data["Passengers+1"] = self.data.loc[:, "Passengers"].shift(-1)
        self.data = self.data[:-2]

    def get_train_inputs(self):
        return self.data.loc[0:107, "Passengers"]

    def get_train_targets(self):
        return self.data.loc[0:107, "Passengers+1"]

    def get_test_inputs(self):
        return self.data.loc[107:142, "Passengers"]

    def get_test_targets(self):
        return self.data.loc[107:142, "Passengers+1"]

    def get_last_train_input(self):
        return self.data.loc[0:107, "Passengers"].tail(1).item()


airlinePassengers = AirlinePassengersDataSet()
airlinePassengers.create_targets()


def create_FFN(inputs, targets):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=8, activation="relu"))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')
    model.fit(inputs, targets, epochs=25, batch_size=1)
    return model


ffn = create_FFN(airlinePassengers.get_train_inputs(), airlinePassengers.get_train_targets())

results = pd.DataFrame()
results["true"] = airlinePassengers.get_test_targets().shift(1)


def validation_forecast(model, inputs):
    predictions = model.predict(inputs)
    return predictions.flatten().tolist()


validation_inputs = airlinePassengers.get_test_inputs()
results["validation"] = validation_forecast(ffn, validation_inputs)


def one_step_ahead_forecast(model, current_value, number_of_predictions):
    one_step_ahead_forecast = list()
    for element in range(0, number_of_predictions):
        prediction = model.predict([current_value])
        one_step_ahead_forecast.append(prediction[0][0])
        current_value = prediction[0]
    return one_step_ahead_forecast


start_value = airlinePassengers.get_last_train_input()
results["one_step_prediction"] = one_step_ahead_forecast(ffn, start_value, len(airlinePassengers.get_test_targets()))

plt.plot(airlinePassengers.get_train_inputs(), color="green", label="training")
plt.plot(results["true"], color="red", label="prediction")
plt.plot(results["validation"], color="blue", label="test")
plt.plot(results["one_step_prediction"], color="orange", label="one_step_prediction")
plt.title("airline passengers prediction")
plt.xlabel("Time[Month]")
plt.ylabel("Passengers[x1000]")
plt.legend(loc="upper left")
plt.savefig("./img/airlinePassengers_keras.png")
plt.show()
