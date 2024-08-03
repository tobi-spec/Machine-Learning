import os
from nixtla import NixtlaClient
from TimeSeries.UnivariatTimeSeriesForecast.AirlinePassengers.airline_passengers_utilities import \
    AirlinePassengersDataSet
from yaml_parser import get_api_key
import matplotlib.pyplot as plt
import pandas as pd

api_key = get_api_key()
nixtla_client = NixtlaClient(api_key=api_key["api-key"])
nixtla_client.validate_api_key()

airline_passengers = AirlinePassengersDataSet()

prediction_results = nixtla_client.forecast(df=airline_passengers.data[:airline_passengers.threshold],
                                            h=35,
                                            time_col='Month',
                                            target_col='Passengers',
                                            freq='MS')

prediction = pd.DataFrame()
prediction["one_step_prediction"] = prediction_results["TimeGPT"]
prediction.index += airline_passengers.threshold

name = "timeGPT"
plt.plot(airline_passengers.passengers, color="red", label="dataset")
plt.plot(airline_passengers.train_data, color="green", label="training")
plt.plot(prediction["one_step_prediction"], color="orange", label="one_step_prediction")
plt.title(f"airline passengers {name}")
plt.xlabel("Time[Month]")
plt.ylabel("Passengers[x1000]")
plt.xticks(range(0, 200, 20))
plt.yticks(range(0, 1000, 100))
plt.legend(loc="upper left")
plt.savefig(f"./airline_passengers_{name}.png")
plt.show()
