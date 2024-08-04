from TimeSeries.UnivariatTimeSeriesForecast.AirlinePassengers.Pytorch.CNN.airline_passengers_pytorch_cnn_model import \
    CNNModel
from TimeSeries.UnivariatTimeSeriesForecast.AirlinePassengers.Pytorch.CNN.airline_passengers_pytorch_cnn_workflow import \
    workflow

if __name__ == "__main__":
    model = CNNModel()
    workflow(model)
