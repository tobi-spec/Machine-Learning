from tabular_data.timeseries_univariat.AirlinePassengers.Pytorch.FF.airline_passengers_pytorch_ff_model import \
    FeedForwardModel
from tabular_data.timeseries_univariat.AirlinePassengers.Pytorch.FF.airline_passengers_pytorch_ff_workflow import \
    workflow

if __name__ == "__main__":
    model = FeedForwardModel()
    workflow(model)