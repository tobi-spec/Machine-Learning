from TimeSeriesPrediction.AnomalieDetection.Electrocardiogram.Keras.AutoEncoder.ecg_keras_autoencoder_model import \
    AutoEncoder
from TimeSeriesPrediction.AnomalieDetection.Electrocardiogram.Keras.AutoEncoder.ecg_keras_autoencoder_workflow import \
    workflow

if __name__ == "__main__":
    model = AutoEncoder()
    workflow(model)
