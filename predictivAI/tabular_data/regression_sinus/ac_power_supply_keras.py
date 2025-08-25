import pandas as pd
from sklearn.model_selection import train_test_split
from keras import Model, layers, optimizers
import matplotlib.pyplot as plt


class ACPowerSupplyData:
    def __init__(self):
        self.data = pd.read_csv("ac_power_supply_data.csv")
        self.population = self.data["Value"]
        self.time = self.data["Time"]


class FeedForwardModel(Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(units=128, activation="relu")
        self.dense2 = layers.Dense(units=256, activation="relu")
        self.dense3 = layers.Dense(units=128, activation="relu")
        self.dense4 = layers.Dense(units=1)

    def call(self, inputs):
        inputs = self.dense1(inputs)
        inputs = self.dense2(inputs)
        inputs = self.dense3(inputs)
        inputs = self.dense4(inputs)
        return inputs


bacterial_growth = ACPowerSupplyData()
x_train, x_test, y_train, y_test = train_test_split(bacterial_growth.time, bacterial_growth.population, test_size=0.25)

model = FeedForwardModel()
model.compile(optimizer=optimizers.SGD(0.001), loss='mean_squared_error')
model.fit(x_train, y_train, epochs=5000, batch_size=3)

prediction = model.predict(x_test)

plt.scatter(x_train, y_train, color="grey")
plt.scatter(x_test, prediction, color="red")
plt.xlabel("population")
plt.ylabel("time")
plt.title('ac power supply regression with Keras')
plt.savefig("sinus_regression_keras.png")
plt.show()




