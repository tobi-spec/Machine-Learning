import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


class AirlinePassengersPreparer:
    def __init__(self):
        self.data = pd.read_csv("../AirlinePassengers.csv", sep=";")
        self.threshold = 107

    def get_train_data(self):
        data = self.data.loc[0:self.threshold, "Passengers"].reset_index(drop=True)
        return pd.DataFrame(data)

    def get_test_data(self):
        data = self.data.loc[self.threshold:142, "Passengers"].reset_index(drop=True)
        return pd.DataFrame(data)


airline_passengers = AirlinePassengersPreparer()
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(airline_passengers.get_train_data())
test = scaler.fit_transform(airline_passengers.get_test_data())


def create_timeseries(data, previous=1):
    inputs, targets = list(), list()
    for i in range(len(data)-previous-1):
        a = data[i: (i+previous), 0]
        inputs.append(a)
        targets.append([data[i + previous, 0]])
    return torch.tensor(inputs, dtype=torch.float), torch.tensor(targets, dtype=torch.float)


lookback = 30
train_inputs, train_targets = create_timeseries(train, lookback)
test_inputs, test_targets = create_timeseries(test, lookback)

train_loader = DataLoader(dataset=TensorDataset(train_inputs, train_targets), batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=TensorDataset(test_inputs, test_targets), batch_size=1, shuffle=False)


class AirlinePassengersModel(nn.Module):
    def __init__(self):
        super().__init__()
        #self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1)
        self.linear1 = nn.Linear(30, 50)
        self.linear2 = nn.Linear(50, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer_function = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, inputs):
        inputs = self.linear1(inputs)
        inputs = self.linear2(inputs)
        return inputs

    def backward(self, train_loader, epoch, num_epochs):
        self.train()
        cumulative_loss = 0

        for x_values, y_values in train_loader:
            prediction = self.forward(x_values)
            loss = self.loss_function(prediction, y_values)
            loss.backward()
            self.optimizer_function.step()
            self.optimizer_function.zero_grad()
            cumulative_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {cumulative_loss / len(train_loader):.4f}")

    def validate(self, val_loader):
        self.eval()
        loss = 0

        with torch.no_grad():
            for x_values, y_values in val_loader:
                prediction = self.forward(x_values)
                loss += self.loss_function(prediction, y_values).item()

        print(f'Validation Loss: {loss / len(val_loader):.4f}')


airline_passenger_model = AirlinePassengersModel()
num_epochs = 200
for epoch in range(num_epochs):
    airline_passenger_model.backward(train_loader, epoch, num_epochs)
    airline_passenger_model.validate(test_loader)


def validation_forecast(model):
    predictions = []
    for element in test_inputs:
        prediction = model(torch.Tensor(element))
        predictions.append(prediction.item())
    return predictions


validation_predictions = validation_forecast(airline_passenger_model)
validation = pd.DataFrame()
validation["true"] = scaler.inverse_transform(test_targets).flatten()
validation["Predictions"] = scaler.inverse_transform([validation_predictions]).flatten()
validation.index += airline_passengers.threshold


def one_step_ahead_forecast(model, current_value, number_of_predictions):
    one_step_ahead_forecast = list()
    for element in range(0, number_of_predictions):
        prediction = model(torch.Tensor(current_value))
        one_step_ahead_forecast.append(prediction.item())
        current_value = np.delete(current_value, 0)
        current_value = np.append(current_value, prediction.item())
        current_value = current_value.reshape(1, 1, current_value.shape[0])
    return one_step_ahead_forecast


start_value = torch.Tensor(test_inputs[0])
number_of_predictions = 40
prediction_results = one_step_ahead_forecast(airline_passenger_model, start_value, number_of_predictions)

prediction = pd.DataFrame()
prediction["one_step_prediction"] = scaler.inverse_transform([prediction_results]).flatten()
prediction.index += airline_passengers.threshold


plt.plot(airline_passengers.get_train_data(), color="green", label="training")
plt.plot(validation["true"], color="red", label="prediction")
plt.plot(validation["Predictions"], color="blue", label="test")
plt.plot(prediction["one_step_prediction"], color="orange", label="one_step_prediction")
plt.title("airline passengers prediction")
plt.xlabel("Time[Month]")
plt.ylabel("Passengers[x1000]")
plt.legend(loc="upper left")
plt.savefig("./airlinePassengers_pytorch.png")
plt.show()
