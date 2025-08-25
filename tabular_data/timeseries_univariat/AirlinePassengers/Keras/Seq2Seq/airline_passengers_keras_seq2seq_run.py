from airline_passengers_keras_seq2seq_workflow import workflow
from airline_passengers_keras_seq2seq_model import seq2seq_model

if __name__ == "__main__":
    model = seq2seq_model()
    workflow(model)
