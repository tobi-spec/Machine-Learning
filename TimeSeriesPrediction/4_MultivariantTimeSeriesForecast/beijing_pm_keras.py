import pandas as pd
from datetime import datetime

dataset = pd.read_csv(
    filepath_or_buffer="BeijingParticulateMatter.csv",
    delimiter=",",
    index_col=0,
    parse_dates=[[1, 2, 3, 4]],
    date_format='%Y %m %d %H')

dataset.drop(["No"], axis=1, inplace=True)
dataset.columns = ['pollution', 'dew', 'temperature', 'pressure', 'wind_direction', 'wind_speed', 'snow', 'rain']
dataset.index.name = 'date'
dataset["pollution"] = dataset.loc[:, "pollution"].fillna(0)
dataset = dataset[24:]
dataset.to_csv("./pollution.csv")


# TODO: Handle first and last results
def last_results(series, range: int):
    inputs = list()
    label = list()
    series.reset_index(drop=True, inplace=True)
    for index, value in series.items():
        inputs.append(series.iloc[index - range:index])
        label.append(value)
    return inputs, label


series = dataset.loc[:, "pollution"]
x, y = last_results(series, 3)
print(x)


# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
