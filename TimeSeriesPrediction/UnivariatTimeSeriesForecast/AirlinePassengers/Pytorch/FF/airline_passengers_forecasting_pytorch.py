import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from TimeSeriesPrediction.UnivariatTimeSeriesForecast.AirlinePassengers.airline_passengers_utilities import *


def main():
    airline_passengers = AirlinePassengersDataSet()
    train = airline_passengers.get_train_data()
    test = airline_passengers.get_test_data()

    lookback: int = 30
    train_timeseries, train_targets = TimeSeriesGenerator(train, lookback).create_timeseries()
    test_timeseries, test_targets = TimeSeriesGenerator(test, lookback).create_timeseries()

    train_timeseries_tensor = torch.tensor(train_timeseries, dtype=torch.float)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float)
    test_timeseries_tensor = torch.tensor(test_timeseries, dtype=torch.float)
    test_targets_tensor = torch.tensor(test_targets, dtype=torch.float)

    train_loader = DataLoader(dataset=TensorDataset(train_timeseries_tensor, train_targets_tensor), batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset=TensorDataset(test_timeseries_tensor, test_targets_tensor), batch_size=1, shuffle=False)

    airline_passenger_model = FeedForwardModel()
    num_epochs = 1000
    for epoch in range(num_epochs):
        airline_passenger_model.backward(train_loader, epoch, num_epochs)
        airline_passenger_model.validate(test_loader)

    validation_results = validation_forecast(airline_passenger_model, test_timeseries_tensor)
    validation = pd.DataFrame()
    validation["validation"] = validation_results
    validation.index += airline_passengers.threshold + lookback

    start_index = -1
    start_value = torch.Tensor(train_timeseries_tensor[start_index])
    number_of_predictions = 80
    prediction_results = one_step_ahead_forecast(airline_passenger_model, start_value, number_of_predictions)

    prediction = pd.DataFrame()
    prediction["one_step_prediction"] = prediction_results
    prediction.index += airline_passengers.threshold + start_index

    plt.plot(airline_passengers.data["Passengers"], color="red", label="dataset")
    plt.plot(airline_passengers.get_train_data(), color="green", label="training")
    plt.plot(validation["validation"], color="blue", label="validation")
    plt.plot(prediction["one_step_prediction"], color="orange", label="one_step_prediction")
    plt.title("airline passengers prediction")
    plt.xlabel("Time[Month]")
    plt.ylabel("Passengers[x1000]")
    plt.xticks(range(0, 200, 20))
    plt.yticks(range(0, 1000, 100))
    plt.legend(loc="upper left")
    plt.savefig("./airlinePassengers_pytorch_ff.png")
    plt.show()


class FeedForwardModel(nn.Module):
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


def validation_forecast(model, inputs):
    predictions = []
    for element in inputs:
        prediction = model(torch.Tensor(element))
        predictions.append(prediction.item())
    return predictions


def one_step_ahead_forecast(model, current_value, number_of_predictions):
    one_step_ahead_forecast = list()
    for element in range(0, number_of_predictions):
        prediction = model(torch.Tensor(current_value))
        one_step_ahead_forecast.append(prediction.item())
        current_value = np.delete(current_value, 0)
        current_value = np.append(current_value, prediction.item())
        current_value = current_value.reshape(1, 1, current_value.shape[0])
    return one_step_ahead_forecast


if __name__ == "__main__":
    main()
