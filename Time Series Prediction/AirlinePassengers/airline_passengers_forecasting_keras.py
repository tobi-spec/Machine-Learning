import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


class AirlinePassengersDataSet:
    def __init__(self):
        self.data = pd.read_csv("./AirlinePassengers.csv", sep=";")
        self.month = self.data["Month"]
        self.passengers = self.data["Passengers"]
        self.passengers_plus_1 = None

    def create_passengers_plus_1(self):
        series1 = self.passengers[1:]
        series2 = pd.Series([0], index=[144])
        self.passengers_plus_1 = pd.concat([series1, series2])

    def add_passengers_plus_1(self):
        self.data["Passengers+1"] = self.passengers_plus_1.reset_index(drop=True)
        self.data.drop(self.data.tail(1).index, inplace=True)

    def get_train_test(self):
        train = self.data[0:107]
        test = self.data[107:142]
        return train, test


timeseries = AirlinePassengersDataSet()
timeseries.create_passengers_plus_1()
timeseries.add_passengers_plus_1()

train, test = timeseries.get_train_test()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=8, activation="relu"))
model.add(tf.keras.layers.Dense(units=1))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')
model.fit(train["Passengers"], train["Passengers+1"], epochs=25, batch_size=1)

predictions = []
for element in test["Passengers"]:
    prediction = model.predict([element])
    predictions.append(prediction[0])
test["Predictions"] = predictions
test.to_csv("./AirlinePassengersTestResults.csv")

plt.plot(train["Month"], train["Passengers"], color="green", label="training")
plt.plot(test["Month"], test["Predictions"], color="red", label="prediction")
plt.plot(test["Month"], test["Passengers"], color="blue", label="test")
plt.title("airline passengers prediction")
plt.xlabel("Time[Month]")
plt.ylabel("Passengers[x1000]")
plt.legend(loc="upper left")
plt.show()
