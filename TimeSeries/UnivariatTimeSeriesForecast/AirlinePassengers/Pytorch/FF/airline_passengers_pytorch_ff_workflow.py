import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from TimeSeries.UnivariatTimeSeriesForecast.AirlinePassengers.airline_passengers_utilities import \
    AirlinePassengersDataSet, TimeSeriesGenerator, plot_results
from yaml_parser import get_hyperparameters


def workflow(model):
    airline_passengers = AirlinePassengersDataSet()
    train = airline_passengers.train_data
    test = airline_passengers.test_data

    hyperparameters: dict = get_hyperparameters("airline_passengers_pytorch_ff_hyperparameter.yaml")

    train_timeseries, train_targets = TimeSeriesGenerator(train, hyperparameters["look_back"],
                                                          hyperparameters["look_out"]).create_timeseries()
    test_timeseries, test_targets = TimeSeriesGenerator(test, hyperparameters["look_back"],
                                                        hyperparameters["look_out"]).create_timeseries()

    train_timeseries_tensor = torch.tensor(train_timeseries, dtype=torch.float)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float)
    test_timeseries_tensor = torch.tensor(test_timeseries, dtype=torch.float)
    test_targets_tensor = torch.tensor(test_targets, dtype=torch.float)

    train_loader = DataLoader(dataset=TensorDataset(train_timeseries_tensor, train_targets_tensor),
                              batch_size=hyperparameters["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=TensorDataset(test_timeseries_tensor, test_targets_tensor),
                             batch_size=hyperparameters["batch_size"], shuffle=True)

    for epoch in range(hyperparameters["epochs"]):
        model.backward(train_loader, epoch, hyperparameters["epochs"])
        model.validate(test_loader)

    validation_results = validation_forecast(model, test_timeseries_tensor)
    validation = pd.DataFrame()
    validation["validation"] = validation_results
    validation.index += airline_passengers.threshold + hyperparameters["look_back"]

    start_index = -1
    start_value = torch.Tensor(train_timeseries_tensor[start_index])
    prediction_results = one_step_ahead_forecast(model, start_value, hyperparameters["number_of_predictions"])

    prediction = pd.DataFrame()
    prediction["one_step_prediction"] = prediction_results
    prediction.index += airline_passengers.threshold + start_index

    plot_results(prediction["one_step_prediction"], validation["validation"], hyperparameters["name"])


def validation_forecast(model, inputs):
    predictions = []
    for element in inputs:
        prediction = model(element)
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
