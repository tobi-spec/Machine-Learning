from TimeSeries.timeseries_anomaly_detection.Electrocardiogram.Keras.AutoEncoder.ecg_keras_autoencoder_model import AutoEncoder
from TimeSeries.timeseries_anomaly_detection.Electrocardiogram.Keras.AutoEncoder.ecg_keras_autoencoder_workflow import workflow

if __name__ == "__main__":
    model = AutoEncoder()
    workflow(model)
