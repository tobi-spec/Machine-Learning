from TimeSeries.UnivariatTimeSeriesForecast.AirlinePassengers.Pytorch.FF.airline_passengers_pytorch_ff_model import \
    FeedForwardModel
from TimeSeries.UnivariatTimeSeriesForecast.AirlinePassengers.Pytorch.FF.airline_passengers_pytorch_ff_workflow import \
    workflow

if __name__ == "__main__":
    model = FeedForwardModel()
    workflow(model)