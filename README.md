# README

This repo is a summary for deep learning / neuronal network practice. 
The different examples are implemented in pytorch, tensorflow keras 
and also a "from scratch" implementation according to the "neuronal networks from scratch" series.

### Linear Regression
basic idea of neuronal networks in keras and pytorch, additionally to a traditionally linear regression model approach for comparison.
All models are used to find the linear relationship between revenue of a ice cream salar and the temperature( data is of course not real world)
All scripts produces images of data points + line of several predictions of the model + run time for easy comparison


### Digit Classification
Classifier for the mnist dataset of handwritten digit. MNIST is implementent in original data files to learn on data preparation.
Digit classification is done in keras and pytorch with and without the use of CNNs. 

## Glossar

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

##Link list:
List of useful videos and articles to learn deep learning

###Basics:
3Blue1Brown Playlist about Deep Learning: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi <br>
Statequest Playlist about Deep Learning: https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1 <br>
Neuronal Networks from Scratch Playlist: https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3 <br>

###Convolutional Neuronal Networks 
https://www.youtube.com/watch?v=YRhxdVk_sIs&t=345s <br>
https://www.youtube.com/watch?v=ZjM_XQa5s6s&t=447s <br>
https://www.youtube.com/watch?v=KuXjwB4LzSA <br>