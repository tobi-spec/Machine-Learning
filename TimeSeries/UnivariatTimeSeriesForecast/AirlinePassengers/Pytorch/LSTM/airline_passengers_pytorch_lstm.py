import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from TimeSeries.UnivariatTimeSeriesForecast.AirlinePassengers.airline_passengers_utilities import \
    AirlinePassengersDataSet, TimeSeriesGenerator

EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 1
LOOK_BACK = 30
LOOK_OUT = 1
PREDICTION_START = -1
NUMBER_OF_PREDICTIONS = 80


def main():
    airline_passengers = AirlinePassengersDataSet()
    train_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = train_scaler.fit_transform(airline_passengers.get_train_data().reshape(-1, 1))

    test_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_test = test_scaler.fit_transform(airline_passengers.get_test_data().reshape(-1, 1))

    train_timeseries, train_targets = TimeSeriesGenerator(scaled_train, LOOK_BACK, LOOK_OUT).create_timeseries()
    test_timeseries, test_targets = TimeSeriesGenerator(scaled_test, LOOK_BACK, LOOK_OUT).create_timeseries()

    train_timeseries = train_timeseries.reshape(train_timeseries.shape[0],train_timeseries.shape[2], train_timeseries.shape[1])
    test_timeseries = test_timeseries.reshape(test_timeseries.shape[0], test_timeseries.shape[2], test_timeseries.shape[1])

    train_timeseries_tensor = torch.tensor(train_timeseries, dtype=torch.float)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float)
    test_timeseries_tensor = torch.tensor(test_timeseries, dtype=torch.float)
    test_targets_tensor = torch.tensor(test_targets, dtype=torch.float)

    train_loader = DataLoader(dataset=TensorDataset(train_timeseries_tensor, train_targets_tensor), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=TensorDataset(test_timeseries_tensor, test_targets_tensor), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel()
    num_epochs = EPOCHS
    for epoch in range(num_epochs):
        model.backward(train_loader, epoch, num_epochs)
        model.validate(test_loader)

    validation_results = validation_forecast(model, test_timeseries_tensor)

    validation = pd.DataFrame()
    validation["validation"] = test_scaler.inverse_transform([validation_results]).flatten()
    validation.index += airline_passengers.threshold + LOOK_BACK

    start_index = PREDICTION_START
    start_value = train_timeseries_tensor[start_index]
    start_value_reshaped = start_value.reshape(1, start_value.shape[0], start_value.shape[1])
    number_of_predictions = NUMBER_OF_PREDICTIONS
    prediction_results = one_step_ahead_forecast(model, start_value_reshaped, number_of_predictions)

    prediction = pd.DataFrame()
    prediction["one_step_prediction"] = train_scaler.inverse_transform([prediction_results]).flatten()
    prediction.index += airline_passengers.threshold + start_index

    plt.plot(airline_passengers.data["Passengers"], color="red", label="dataset")
    plt.plot(airline_passengers.get_train_data(), color="green", label="training")
    plt.plot(validation["validation"], color="blue", label="validation")
    plt.plot(prediction["one_step_prediction"], color="orange", label="one_step_prediction")
    plt.title("airline passengers prediction LSTM")
    plt.xlabel("Time[Month]")
    plt.ylabel("Passengers[x1000]")
    plt.xticks(range(0, 200, 20))
    plt.yticks(range(0, 1000, 100))
    plt.legend(loc="upper left")
    plt.savefig("./airlinePassengers_pytorch_lstm.png")
    plt.show()


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=30, hidden_size=50, num_layers=1, batch_first=True)
        self.linear1 = nn.Linear(50, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer_function = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def forward(self, inputs):
        x, _ = self.lstm(inputs)
        x = x[:, -1, :]
        x = self.linear1(x)
        return x

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
        prediction = model.forward(element[None, :, :])
        predictions.append(prediction.item())
    return predictions


def one_step_ahead_forecast(model, current_value, number_of_predictions):
    one_step_ahead_forecast = list()
    for element in range(0, number_of_predictions):
        prediction = model.forward(torch.Tensor(current_value))
        one_step_ahead_forecast.append(prediction.item())
        current_value = np.delete(current_value, 0)
        current_value = np.append(current_value, prediction.detach().numpy())
        current_value = current_value.reshape(1, 1, current_value.shape[0])
    return one_step_ahead_forecast


if __name__ == "__main__":
    main()
