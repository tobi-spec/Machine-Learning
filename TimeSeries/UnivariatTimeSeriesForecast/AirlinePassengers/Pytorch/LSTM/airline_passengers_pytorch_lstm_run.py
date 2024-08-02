from TimeSeries.UnivariatTimeSeriesForecast.AirlinePassengers.Pytorch.LSTM.airline_passengers_pytorch_lstm_model import \
    LSTMModel
from TimeSeries.UnivariatTimeSeriesForecast.AirlinePassengers.Pytorch.LSTM.airline_passengers_pytorch_lstm_workflow import \
    workflow

if __name__ == "__main__":
    model = LSTMModel()
    workflow(model)