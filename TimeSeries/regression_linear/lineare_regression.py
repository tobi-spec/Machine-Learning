import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import timeit

# train a linear Regression model - most basic machine learning algorithm.
# make predictions about linear correlation between temperature and ice cream revenue
start = timeit.default_timer()
IceCream = pd.read_csv("IceCreamData.csv")

x_values = IceCream.loc[:, "Temperature"].to_frame()
y_values = IceCream.loc[:, "Revenue"]

x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.25)

regressor = LinearRegression(fit_intercept=True)
regressor.fit(x_train, y_train)

print('Linear Model Coefficient (m): ', regressor.coef_)
print('Linear Model Coefficient (b): ', regressor.intercept_)

y_predict = regressor.predict(x_train)
stop = timeit.default_timer()

plt.scatter(x_train, y_train, color="grey")
plt.plot(x_train, y_predict, color="red")
plt.xlabel("temperature [degC]")
plt.ylabel("revenue [dollars]")
plt.title('normal linear regression')
plt.figtext(0.2, 0.8, f"run time[s]: {stop-start}")
plt.savefig("./img/linear_regression")
plt.show()

