# README

This repository is a summary for deep learning / neuronal network examples. 
The different examples are implemented in pytorch and tensorflow keras. It exists also a basic "from scratch" implementation 
according to the "neuronal networks from scratch" series.

The examples are divided into the 3 main topics of neuronal networks: time series, natural language processing, and computer vision 


## Time Series

### Linear Regression
Most basic implementation of a neuronal network - a linear equation. 
This is used for a simple regression to predict revenue of a ice cream truck for a certain temperature

### Univariat Time Series Forecast
Forecasting of a time series model with one attribute. The data comes from the airline passengers dataset. Aim is to predict 
the progression of passenger numbers. The forecast is made by a one-step-ahead prediction. To get in touch with different 
models and layers the forecast is implement with following variations:

- feed forward
- lstm
- rnn
- cnn
- seq2seq
- attention

Also two pretrained transformer models are used: timeGPT and Darts. As well as a funky approach by using the chatGPT API 
to ask for a forecast via prompt. 

### Multivariat Time Series Forecast
Forecast of a time series model with multiple attributes. The data are the beijing pm2.5 dataset (pm stands for particular matter). 
Aim is to predict the air pollution by multiple factors. The forecast is made by a one-step-ahead prediction. Currently there is only one
implementation with a keras feed forward model for a basic approach. 

### Anomaly Detection
...

## Computervision

### Image Classification
Classifier for detection of handwritten digits from 0 to 9. The MNIST dataset for handwritten digits is used as data source.
MNIST contains 60.000 images and labels for training and 10.000 images and labels for test, each image is 28x28 pixels and in b/w.
Currently, implemented are models based of feed forward layers and cnn layers, both in keras and in pytorch. 


## Natural Language Processing
...


