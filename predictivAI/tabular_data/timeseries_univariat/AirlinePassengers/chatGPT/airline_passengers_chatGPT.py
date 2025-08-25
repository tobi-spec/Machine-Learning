from predictivAI.tabular_data.timeseries_univariat.AirlinePassengers.airline_passengers_utilities import \
    AirlinePassengersDataSet
from yaml_parser import get_api_key
from openai import OpenAI
import json
import matplotlib.pyplot as plt
import pandas as pd

api_key = get_api_key()
client = OpenAI(
    api_key=api_key["openAI-api-key"]
)

airline_passengers = AirlinePassengersDataSet()
train = airline_passengers.data[:airline_passengers.threshold]

horizon = 30
prompt = f"""Given the dataset delimited by the triple backticks, forecast next {horizon} values of the time series.
                Return the answer in JSON format, containing two keys: 'Period' and 'Forecast', and list of values assigned to them. 
                Return only the forecasts, not the Python code or additional text. ``` {train}``` """

model = "gpt-4o"
chat_completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system",
         "content": "You are experienced in data analysis"},
        {"role": "user", "content": prompt}
    ]
)

model_response = chat_completion.choices[0].message.content[7:-4]
valid_json_response = json.loads(model_response)
prediction = pd.DataFrame.from_records(valid_json_response)
prediction.index += airline_passengers.threshold

plt.plot(airline_passengers.passengers, color="red", label="dataset")
plt.plot(airline_passengers.train_data, color="green", label="training")
plt.plot(prediction["Forecast"], color="orange", label="one_step_prediction")
plt.title(f"airline passengers {model}")
plt.xlabel("Time[Month]")
plt.ylabel("Passengers[x1000]")
plt.xticks(range(0, 200, 20))
plt.yticks(range(0, 1000, 100))
plt.legend(loc="upper left")
plt.savefig(f"./airline_passengers_{model}.png")
plt.show()