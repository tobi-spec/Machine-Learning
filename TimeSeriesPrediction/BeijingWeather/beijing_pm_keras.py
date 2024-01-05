import pandas as pd
from datetime import datetime

dataset = pd.read_csv(
    "BeijingParticulateMatter.csv",
    delimiter=";",
    parse_dates=True,
    date_format='%Y %m %d %H')

print(dataset)

#https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
#https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler