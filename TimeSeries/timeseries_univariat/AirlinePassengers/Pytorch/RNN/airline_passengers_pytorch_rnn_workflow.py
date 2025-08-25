import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from TimeSeries.timeseries_univariat.AirlinePassengers.airline_passengers_utilities import \
    AirlinePassengersDataSet, TimeSeriesGenerator, plot_results
from yaml_parser import get_hyperparameters


def workflow(model):
    airline_passengers = AirlinePassengersDataSet()
    train_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = train_scaler.fit_transform(airline_passengers.train_data.reshape(-1, 1))

    test_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_test = test_scaler.fit_transform(airline_passengers.test_data.reshape(-1, 1))

    hyperparameters: dict = get_hyperparameters("airline_passengers_pytorch_rnn_hyperparameter.yaml")

    train_timeseries, train_targets = TimeSeriesGenerator(scaled_train, hyperparameters["look_back"],
                                                          hyperparameters["look_out"]).create_timeseries()
    test_timeseries, test_targets = TimeSeriesGenerator(scaled_test, hyperparameters["look_back"],
                                                          hyperparameters["look_out"]).create_timeseries()

    train_timeseries = train_timeseries.reshape(train_timeseries.shape[0],train_timeseries.shape[2], train_timeseries.shape[1])
    test_timeseries = test_timeseries.reshape(test_timeseries.shape[0], test_timeseries.shape[2], test_timeseries.shape[1])

    train_timeseries_tensor = torch.tensor(train_timeseries, dtype=torch.float)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float)
    test_timeseries_tensor = torch.tensor(test_timeseries, dtype=torch.float)
    test_targets_tensor = torch.tensor(test_targets, dtype=torch.float)

    train_loader = DataLoader(dataset=TensorDataset(train_timeseries_tensor, train_targets_tensor), batch_size=hyperparameters["batch_size"], shuffle=False)
    test_loader = DataLoader(dataset=TensorDataset(test_timeseries_tensor, test_targets_tensor), batch_size=hyperparameters["batch_size"], shuffle=False)

    for epoch in range(hyperparameters["epochs"]):
        model.backward(train_loader, epoch, hyperparameters["epochs"])
        model.validate(test_loader)

    validation_results = validation_forecast(model, test_timeseries_tensor)

    validation = pd.DataFrame()
    validation["validation"] = test_scaler.inverse_transform([validation_results]).flatten()
    validation.index += airline_passengers.threshold + hyperparameters["look_back"]

    start_index = -1
    start_value = train_timeseries_tensor[start_index]
    start_value_reshaped = start_value.reshape(1, start_value.shape[0], start_value.shape[1])
    number_of_predictions = hyperparameters["number_of_predictions"]
    prediction_results = one_step_ahead_forecast(model, start_value_reshaped, number_of_predictions)

    prediction = pd.DataFrame()
    prediction["one_step_prediction"] = train_scaler.inverse_transform([prediction_results]).flatten()
    prediction.index += airline_passengers.threshold + start_index

    plot_results(prediction["one_step_prediction"], validation["validation"], hyperparameters["name"])


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
