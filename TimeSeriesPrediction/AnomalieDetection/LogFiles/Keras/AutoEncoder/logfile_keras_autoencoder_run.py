from TimeSeriesPrediction.AnomalieDetection.LogFiles.Keras.AutoEncoder.logfile_keras_autoencoder_model import \
    AutoEncoder
from TimeSeriesPrediction.AnomalieDetection.LogFiles.Keras.AutoEncoder.logfile_keras_autoencoder_workflow import \
    workflow

if __name__ == "__main__":
    model = AutoEncoder()
    workflow(model)