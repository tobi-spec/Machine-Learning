from enum import Enum


class NeuronalNetworkTypes(Enum):
    FEED_FORWARD: str = "feedforward"
    LSTM: str = "lstm"
    ATTENTION: str = "attention"
    CNN: str = "cnn"
    RNN: str = "rnn"
