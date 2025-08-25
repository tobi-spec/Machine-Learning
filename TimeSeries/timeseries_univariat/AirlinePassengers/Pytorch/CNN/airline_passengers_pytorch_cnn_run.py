from TimeSeries.timeseries_univariat.AirlinePassengers.Pytorch.CNN.airline_passengers_pytorch_cnn_model import \
    CNNModel
from TimeSeries.timeseries_univariat.AirlinePassengers.Pytorch.CNN.airline_passengers_pytorch_cnn_workflow import \
    workflow

if __name__ == "__main__":
    model = CNNModel()
    workflow(model)
