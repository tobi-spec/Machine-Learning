import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import timeit

start = timeit.default_timer()

IceCream = pd.read_csv("IceCreamData.csv")
x_values = IceCream[["Temperature"]]
y_values = IceCream["Revenue"]
x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.25)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1,
                                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                bias_initializer=tf.keras.initializers.Zeros())
          )
model.compile(optimizer=tf.keras.optimizers.Adam(0.5), loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=1)

x_calculate = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
y_prediction = []
for temperatur in x_calculate:
    prediction = model.predict([temperatur])
    y_prediction.append(prediction[0])
stop = timeit.default_timer()

plt.scatter(x_train, y_train, color="grey")
plt.plot(x_calculate, y_prediction, color="red")
plt.xlabel("temperature [degC]")
plt.ylabel("revenue [dollars]")
plt.title('Linear regression with Keras')
plt.figtext(0.2, 0.8, f"run time[s]: {stop-start}")
plt.savefig("./img/linear_regression_keras")
plt.show()