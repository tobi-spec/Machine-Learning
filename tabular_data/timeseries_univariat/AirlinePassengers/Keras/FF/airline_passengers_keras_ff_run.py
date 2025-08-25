from airline_passengers_keras_ff_workflow import workflow
from airline_passengers_keras_ff_model import FeedForwardModel


if __name__ == "__main__":
    model = FeedForwardModel()
    workflow(model)
    