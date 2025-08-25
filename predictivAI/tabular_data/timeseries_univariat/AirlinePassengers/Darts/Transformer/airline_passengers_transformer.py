import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel
from darts.metrics import mape
from darts.datasets import AirPassengersDataset


series = AirPassengersDataset().load().astype(np.float32)
train, val = series.split_after(pd.Timestamp("19590101"))

scaler = Scaler()
train_scaled = scaler.fit_transform(train)
val_scaled = scaler.transform(val)
series_scaled = scaler.transform(series)


my_model = TransformerModel(
    input_chunk_length=12,
    output_chunk_length=1,
    batch_size=3,
    n_epochs=400,
    nr_epochs_val_period=10,
    d_model=16,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
    activation="relu",
    random_state=42,
    force_reset=True,
)

my_model.fit(series=train_scaled, val_series=val_scaled, verbose=True)

def eval_model(model, n, series, val_series):
    pred_series = model.predict(n=n)
    plt.figure(figsize=(8, 5))
    series.plot(label="actual")
    pred_series.plot(label="forecast")
    plt.title("MAPE: {:.2f}%".format(mape(pred_series, val_series)))
    plt.legend()
    plt.savefig("airlinePassengers_darts_transformer")
    plt.show()


eval_model(my_model, 60, series_scaled, val_scaled)