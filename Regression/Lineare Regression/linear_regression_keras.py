import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# train a neuronal network with single neuron -> linear.
# make predictions about linear correlation between temperature and ice cream revenue

IceCream = pd.read_csv("IceCreamData.csv")

y_values = IceCream["Revenue"]
x_values = IceCream[["Temperature"]]

x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.25)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1))

model.compile(optimizer=tf.keras.optimizers.Adam(0.5), loss='mean_squared_error')
epochs_hist = model.fit(x_train, y_train, epochs = 100)

x_calculate = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
y_prediction = []
for temperatur in x_calculate:
    prediction = model.predict([temperatur])
    y_prediction.append(prediction[0])

plt.scatter(x_train, y_train, color="grey")
plt.plot(x_calculate, y_prediction, color="red")
plt.xlabel("revenue [dolars]")
plt.ylabel("temperature [degC]")
plt.title('Revenue Generated vs. Temperature for Ice Cream Stand')
plt.show()