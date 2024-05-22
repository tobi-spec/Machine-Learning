from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from TimeSeriesPrediction.UnivariatTimeSeriesForecast.AirlinePassengers.Keras.workflows.airline_passengers_keras_seq2seq_workflow import workflow


def seq2seq_model():
    encoder_input = Input(shape=(1, 30))
    encoder_lstm, state_h, state_c = LSTM(50, return_state=True)(encoder_input)
    encoder_states = [state_h, state_c]

    decoder_input = Input(shape=(1, 1))
    decoder_lstm = LSTM(50, return_sequences=True, return_state=True)
    decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
    decoder_output = Dense(1)(decoder_output)

    return Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)


if __name__ == "__main__":
    model = seq2seq_model()
    workflow(model, "seq2seq")
