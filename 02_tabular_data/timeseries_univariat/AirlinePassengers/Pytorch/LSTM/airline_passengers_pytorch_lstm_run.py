from tabular_data.timeseries_univariat.AirlinePassengers.Pytorch.LSTM.airline_passengers_pytorch_lstm_model import \
    LSTMModel
from tabular_data.timeseries_univariat.AirlinePassengers.Pytorch.LSTM.airline_passengers_pytorch_lstm_workflow import \
    workflow

if __name__ == "__main__":
    model = LSTMModel()
    workflow(model)