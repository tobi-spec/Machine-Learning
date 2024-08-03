from TimeSeries.UnivariatTimeSeriesForecast.AirlinePassengers.Pytorch.RNN.airline_passengers_pytorch_rnn_model import \
    RNNModel
from TimeSeries.UnivariatTimeSeriesForecast.AirlinePassengers.Pytorch.RNN.airline_passengers_pytorch_rnn_workflow import \
    workflow

if __name__ == "__main__":
    model = RNNModel()
    workflow(model)