import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tabular_data.timeseries_univariat.AirlinePassengers.airline_passengers_utilities import \
    AirlinePassengersDataSet, TimeSeriesGenerator, plot_results, PytorchForecaster
from neuronal_network_types import NeuronalNetworkTypes
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
    prediction_results = PytorchForecaster(model,
                                           start_value,
                                           hyperparameters["number_of_predictions"],
                                           NeuronalNetworkTypes.FEED_FORWARD).one_step_ahead_forecast()

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
