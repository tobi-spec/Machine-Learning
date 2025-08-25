from tabular_data.timeseries_anomaly_detection.LogFiles.Keras.AutoEncoder.logfile_keras_autoencoder_model import AutoEncoder
from tabular_data.timeseries_anomaly_detection.LogFiles.Keras.AutoEncoder.logfile_keras_autoencoder_workflow import workflow

if __name__ == "__main__":
    model = AutoEncoder()
    workflow(model)