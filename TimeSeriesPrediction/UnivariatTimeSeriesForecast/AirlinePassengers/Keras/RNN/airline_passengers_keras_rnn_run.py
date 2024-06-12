from airline_passengers_keras_rnn_workflow import workflow
from airline_passengers_keras_rnn_model import RNNModel

if __name__ == "__main__":
    rnn_model = RNNModel()
    workflow(rnn_model)