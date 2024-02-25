import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import timeit

start = timeit.default_timer()


class IceCreamData:
    def __init__(self):
        self.data = pd.read_csv("IceCreamData.csv")

    def get_temperature(self):
        return self.data.loc[:, "Temperature"]

    def get_revenue(self):
        return self.data.loc[:, "Revenue"]


iceCreamData = IceCreamData()
x_values = iceCreamData.get_temperature()
y_values = iceCreamData.get_revenue()
x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.25)


def create_linear_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=1,
                                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                    bias_initializer=tf.keras.initializers.Zeros())
              )
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mean_squared_error')
    return model


linear_model = create_linear_model()
linear_model.fit(x_train, y_train, epochs=20, batch_size=1)


def validation_prediction():
    return linear_model.predict([x_test])


validation_results = validation_prediction()

stop = timeit.default_timer()

plt.scatter(x_train, y_train, color="grey")
plt.scatter(x_test, validation_results, color="red")
plt.xlabel("temperature [degC]")
plt.ylabel("revenue [dollars]")
plt.title('Linear regression with Keras')
plt.figtext(0.2, 0.8, f"run time[s]: {stop-start}")
plt.savefig("./img/linear_regression_keras")
plt.show()
