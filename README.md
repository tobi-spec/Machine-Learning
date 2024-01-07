# README

This repo is a summary for deep learning / neuronal network practice. 
The different examples are implemented in pytorch, tensorflow keras 
and also a "from scratch" implementation according to the "neuronal networks from scratch" series.

## Overview

### Time Series Forecast
Predicting future values based on the analysis of past data points in a chronological sequence. 
It is commonly used to anticipate trends, patterns, or behaviors in time-ordered datasets

**IceCreamRevenue**: Basic idea of neuronal networks, additionally to a traditionally linear regression model approach for comparison.
All models are used to find the linear relationship between revenue of a ice cream salar and the temperature( data is of course not real world)

**AirlinePassengers**: Predictor for airline passenger forecast, simple neuronal network for basic understanding timeseries forecasting. 

### Computervision
Extract, process, and comprehend information from visual data, typically in the form of images or videos.
Involves image recognition, object detection, and scene understanding.

**Digit Classification**: Classifier for the mnist dataset of handwritten digit. MNIST is implementent in original data files to learn on data preparation.
Done with and without the use of CNNs.

### Natural Language Processing
Giving computers the ability to support and manipulate human language. 
Involve speech recognition, natural-language understanding, and natural-language generation.


## Glossar

**Epoch**: One forward pass and one backward pass of all the training examples. <br>

**Batch**: Since one epoch is too big to feed to the computer at once, we divide it in serveral smaller batches. 
After every batch backward pass and After one epoch the weights decent is calculated<br>

**Batch Size**: Total number of traninf examples in a single batch <br>

**Iteration**: Number of batches needed to complete one Epoch

### Layer weight initializers
pytorch: https://pytorch.org/docs/stable/nn.init.html <br>
keras: https://keras.io/api/layers/initializers/ <br>

### Loss Functions
pytorch: https://pytorch.org/docs/stable/nn.html#loss-functions <br>
keras: https://keras.io/api/losses/ <br>
 
loss functions are divided in regression losses like Mean Error functions and classification losses like cross entropy functions

#### Good to know
 - (Keras) Binary Cross Entropy - two Labels 
 - (Keras) Categorical Cross Entropy - one-hot encoded labels
 - (Keras) Sparse Categorical Cross Entropy - labels with logits/ pure numbers
 - (Pytorch) CrossEntropyLoss - equivalent to applying LogSoftmax on an input, followed by NLLLoss -> no softmax in output layer!

### Optimizer Functions
pytorch: https://pytorch.org/docs/stable/optim.html <br>
keras: https://keras.io/api/optimizers/ <br>

- Momentum: Running Average of the last x gradients to overcome local minima
- Learning rate decay: Makes learning steps small and small to not overstep global minimum

## Link list:
List of useful videos and articles to learn deep learning

### Basics:
3Blue1Brown Playlist about Deep Learning: <br> https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi <br>
Statequest Playlist about Deep Learning: <br> https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1 <br>
Neuronal Networks from Scratch Playlist: <br> https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3 <br>

### Convolutional Neuronal Networks 
https://www.youtube.com/watch?v=YRhxdVk_sIs&t=345s <br>
https://www.youtube.com/watch?v=ZjM_XQa5s6s&t=447s <br>
https://www.youtube.com/watch?v=KuXjwB4LzSA <br>

### Recurrent Neuronal Networks
https://www.youtube.com/watch?v=LHXXI4-IEns