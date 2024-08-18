import pandas as pd
from sklearn.model_selection import train_test_split
from keras import Model, layers, optimizers
import matplotlib.pyplot as plt


class BacteriaGrowth:
    def __init__(self):
        self.data = pd.read_csv("./bacteria_growth_data.csv")
        self.population = self.data["Population"]
        self.time = self.data["Time"]


class FeedForwardModel(Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(units=15, activation="sigmoid")
        self.dense2 = layers.Dense(units=15, activation="softplus")
        self.dense3 = layers.Dense(units=1)

    def call(self, inputs):
        inputs = self.dense1(inputs)
        inputs = self.dense2(inputs)
        inputs = self.dense3(inputs)
        return inputs


bacterial_growth = BacteriaGrowth()
x_train, x_test, y_train, y_test = train_test_split(bacterial_growth.time, bacterial_growth.population, test_size=0.25)

model = FeedForwardModel()
model.compile(optimizer=optimizers.Adam(0.001), loss='mean_squared_error')
model.fit(x_train, y_train, epochs=150, batch_size=1)

prediction = model.predict(x_test)

plt.scatter(x_train, y_train, color="grey")
plt.scatter(x_test, prediction, color="red")
plt.xlabel("population")
plt.ylabel("time")
plt.title('exponential regression with Keras')
plt.savefig("exponential_regression_keras")
plt.show()




