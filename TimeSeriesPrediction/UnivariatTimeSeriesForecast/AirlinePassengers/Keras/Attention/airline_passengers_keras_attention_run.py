from airline_passengers_keras_attention_model import EncoderModel
from airline_passengers_keras_attention_workflow import workflow

if __name__ == "__main__":
    encoder_model = EncoderModel()
    workflow(encoder_model)