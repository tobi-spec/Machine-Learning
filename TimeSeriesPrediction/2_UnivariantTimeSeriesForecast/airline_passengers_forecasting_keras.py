import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


class AirlinePassengersDataSet:
    def __init__(self):
        self.data = pd.read_csv("./AirlinePassengers.csv", sep=";")
        self.month = self.data.loc[:, "Month"]
        self.passengers = self.data.loc[:, "Passengers"]
        self.passengers_plus_1 = None

    def create_forecast_sequence(self):
        self.passengers_plus_1 = self.passengers.shift(-1)
        self.passengers_plus_1 = self.passengers_plus_1.iloc[:-1]
        self.data["Passengers+1"] = self.passengers_plus_1
        self.passengers = self.passengers.iloc[:-1]

    def get_train_test(self):
        train = self.data[0:107]
        test = self.data[107:142]
        return train, test


timeseries = AirlinePassengersDataSet()
timeseries.create_forecast_sequence()
train, test = timeseries.get_train_test()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=8, activation="relu"))
model.add(tf.keras.layers.Dense(units=1))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')
model.fit(train.loc[:, "Passengers"], train.loc[:, "Passengers+1"], epochs=25, batch_size=1)

predictions = model.predict(test.loc[:, "Passengers"])
test["Predictions"] = predictions
test.to_csv("./AirlinePassengersResultsKeras.csv")

plt.plot(train["Month"], train["Passengers"], color="green", label="training")
plt.plot(test["Month"], test["Predictions"], color="red", label="prediction")
plt.plot(test["Month"], test["Passengers"], color="blue", label="test")
plt.title("airline passengers prediction")
plt.xlabel("Time[Month]")
plt.ylabel("Passengers[x1000]")
plt.legend(loc="upper left")
plt.savefig("./img/airlinePassengers_keras.png")
plt.show()
