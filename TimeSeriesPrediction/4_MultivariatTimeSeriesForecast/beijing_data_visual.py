import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(
    filepath_or_buffer="BeijingParticulateMatter.csv",
    delimiter=",",
    index_col=0,
    parse_dates=[[1, 2, 3, 4]],
    date_format='%Y %m %d %H')

dataset.drop(["No"], axis=1, inplace=True)
dataset.columns = ['pollution', 'dew', 'temperature', 'pressure', 'wind_direction', 'wind_speed', 'snow', 'rain']
dataset.index.name = 'date'
dataset.loc[:, "pollution"].fillna(0, inplace=True)
dataset = dataset[24:]

fig, ax = plt.subplots(3,3)
ax[0, 0].plot(dataset['rain'])
ax[0, 0].set_title('rain')

ax[0, 1].plot(dataset["dew"])
ax[0, 1].set_title("Dew")

ax[0, 2].plot(dataset['temperature'])
ax[0, 2].set_title('Temperature')

ax[1, 0].plot(dataset['pressure'])
ax[1, 0].set_title('Pressure')

ax[1, 1].plot(dataset['wind_speed'])
ax[1, 1].set_title('wind_speed')

ax[1, 2].plot(dataset['snow'])
ax[1, 2].set_title('Snow')

ax[2, 1].plot(dataset["pollution"])
ax[2, 1].set_title("pollution")

plt.show()