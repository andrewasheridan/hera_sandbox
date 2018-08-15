# CNN_DS_BN_BC

import sys
import numpy as np
import tensorflow as tf
from CNN_DS_BN_C import CNN_DS_BN_C 

class CNN_DS_BN_BC(CNN_DS_BN_C):
    """ CNN_DS_BN_BC

    Downsampling binary classifier
    CNN: Convolutional Neural Network.
    DS: DownSampling. Each convolution is followed by a downsampling convolution
    BN: All non-linearalities have batch-normalization applied.
    BC: Binary Classification, this network classifies samples of having one of two labels.

    Each layer starts with a 1x3 convolution with 2**i filters (i = layer index) with stride of 1.
    This is fed into a 1x5 convolution with the same number of filters, but stride of 2 (50% downsample)

    Leaky_ReLU and batch normalization is applied after each convolution. Downsamples convolutions have dropout.
    Biases are added before activations.

    Output of last layer is fed to fully connected layer (with no activation)

    Cost function is softmax cross entropy

    """
    __doc__ += CNN_DS_BN_C.__doc__
    def __init__(self,
                 name,
                 num_downsamples,
                 log_dir = '../logs/',
                 dtype = tf.float32,
                 adam_initial_learning_rate = 0.0001,
                 verbose = True):
        
         CNN_DS_BN_C.__init__(self,
                              name=name,
                              num_downsamples=num_downsamples,
                              num_classes=2,
                              log_dir=log_dir,
                              dtype=dtype,
                              adam_initial_learning_rate=adam_initial_learning_rate,
                              verbose=verbose)
 
