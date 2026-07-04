from airline_passengers_keras_lstm_workflow import workflow
from airline_passengers_keras_lstm_model import LSTMModel

if __name__ == "__main__":
    lstm_model = LSTMModel()
    workflow(lstm_model)
