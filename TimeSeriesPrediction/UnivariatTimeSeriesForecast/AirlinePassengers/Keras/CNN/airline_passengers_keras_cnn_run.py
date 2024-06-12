from airline_passengers_keras_cnn_model import CNNModel
from airline_passengers_keras_cnn_workflow import workflow

if __name__ == "__main__":
    cnn_model = CNNModel()
    workflow(cnn_model, "cnn")
