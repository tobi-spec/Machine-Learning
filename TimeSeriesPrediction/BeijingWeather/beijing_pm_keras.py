import pandas as pd
from datetime import datetime

dataset = pd.read_csv(
    filepath_or_buffer="BeijingParticulateMatter.csv",
    delimiter=",",
    index_col=0,
    parse_dates=[[1 ,2 , 3, 4]],
    date_format='%Y %m %d %H')

dataset.drop(["No"], axis=1, inplace=True)
dataset.columns = ['pollution', 'dew', 'temperature', 'pressure', 'wind_direction', 'wind_speed', 'snow', 'rain']
dataset.index.name = 'date'
dataset["pollution"] = dataset["pollution"].fillna(0)
dataset = dataset[24:]
dataset.to_csv("./pollution.csv")
print(dataset)

#https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
#https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
