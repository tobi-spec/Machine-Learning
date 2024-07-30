from enum import Enum


class NeuronalNetworkTypes(Enum(str)):
    FEED_FORWARD: str = "feedforward"
    LSTM: str = "lstm"
    ATTENTION: str = "attention"
    CNN: str = "cnn"
    RNN: str = "rnn"
