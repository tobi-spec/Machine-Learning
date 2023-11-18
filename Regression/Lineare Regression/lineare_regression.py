import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# train a linear Regression model - most basic machine learning algorithm.
# make predictions about linear correlation between temperature and ice cream revenue

IceCream = pd.read_csv("IceCreamData.csv")

y_values = IceCream["Revenue"]
x_values = IceCream[["Temperature"]]

x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.25)

print(x_train.shape)
print(y_train.shape)

regressor = LinearRegression(fit_intercept=True)
regressor.fit(x_train, y_train)

print('Linear Model Coefficient (m): ', regressor.coef_)
print('Linear Model Coefficient (b): ', regressor.intercept_)

y_predict = regressor.predict(x_train)

plt.scatter(x_train, y_train, color="grey")
plt.plot(x_train, y_predict, color="red")
plt.xlabel("revenue [dolars]")
plt.ylabel("temperature [degC]")
plt.title('Revenue Generated vs. Temperature for Ice Cream Stand')
plt.show()

