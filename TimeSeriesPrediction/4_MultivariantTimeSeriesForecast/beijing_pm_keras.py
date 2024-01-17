import pandas as pd

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

class TimeSeriesConstructor:
    def __init__(self, series):
        self.dataframe = pd.DataFrame(data=series)

    def create_sequences(self, span: int):
        inputs = list()
        series.reset_index(drop=True, inplace=True)
        for index, value in series.items():
            inputs.append(series.iloc[index - span:index].to_list())
        self.dataframe["inputs"] = inputs

    def drop_empty_sequences(self):
        self.dataframe = self.dataframe[self.dataframe.inputs.str.len() != 0]


series = dataset.loc[:, "pollution"]
timeSeries = TimeSeriesConstructor(series)
timeSeries.create_sequences(span=3)
timeSeries.drop_empty_sequences()
print(timeSeries.dataframe)



# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
