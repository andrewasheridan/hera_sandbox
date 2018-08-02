# CNN_DS_BN_C

import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '../modules'))

from Restoreable_Component import Restoreable_Component

import tensorflow as tf
import numpy as np

class CNN_DS_BN_C(Restoreable_Component):
    """ CNN: Convolutional Neural Network.
        DS: DownSampling. Each convolution is followed by a downsampling convolution
        BN: All non-linearalities have batch-normalization applied.
        C: Classification, this network classifies samples as having one of many labels.
 
        Each layer starts with a 1*N convolution with 4**(i+1) filters (i = layer index) with stride of 1.
            - N = max([layer.get_shape().as_list()[2] / 4, 2])
            - not sure why i changed this from just '3'

        This is fed into a 1xW convolution with the same number of filters, but stride of 2 (50% downsample)
            - W = N * 2
            - also. not sure why this isnt just 5.. experimenting i guess
            
        Leaky_ReLU and batch normalization is applied after each convolution. Downsamples convolutions have dropout.
        Biases are added before activations.

        Output of last layer is fed to fully connected layer (with no activation)

        Cost function is softmax cross entropy

       """

    def __init__(self,
                 name,
                 num_downsamples,
                 num_classes,
                 log_dir = 'logs/',
                 dtype = tf.float32,
                 adam_initial_learning_rate = 0.0001,
                 verbose = True):
    
        Restoreable_Component.__init__(self, name=name, log_dir=log_dir, verbose=verbose)
                
        self.num_downsamples = num_downsamples
        
        self.dtype = dtype
        self.adam_initial_learning_rate = adam_initial_learning_rate
        

        self._num_freq_channels = 1024
        self._layers = []
        self.num_classes = num_classes #  classifier...


    def create_graph(self):

        self.save_params()
        self._msg = '\rcreating network graph '; self._vprint(self._msg)

        
        tf.reset_default_graph()
        self.is_training = tf.placeholder(dtype = tf.bool, shape = [], name = 'is_training')
        
        with tf.variable_scope('keep_probs'):
            
            self._msg += '.'; self._vprint(self._msg)

            self.sample_keep_prob = tf.placeholder(self.dtype, name = 'sample_keep_prob')
            self.downsample_keep_prob = tf.placeholder(self.dtype, name = 'downsample_keep_prob')
        
        
        with tf.variable_scope('sample'):
            
            self._msg += '.'; self._vprint(self._msg)
            
            self.X = tf.placeholder(self.dtype, shape = [None, 1, self._num_freq_channels, 1], name = 'X')
            self.X = tf.nn.dropout(self.X, self.sample_keep_prob)
            
        trainable = lambda shape, name : tf.get_variable(name = name,
                                                         dtype = self.dtype,
                                                         shape = shape,
                                                         initializer = tf.contrib.layers.xavier_initializer())


        for i in range(self.num_downsamples):
            self._msg += '.'; self._vprint(self._msg)
            
            with tf.variable_scope('conv_layer_{}'.format(i)):

                layer = self.X if i == 0 else self._layers[-1]
                
                # filter shape:
                fh = 1 # filter height = 1 for 1D convolution
                fw = 3#1 + np.max([layer.get_shape().as_list()[2] / 32, 2])
                fic = 4**(i) # num in channels = number of incoming filters
                foc = 4**(i+1) # num out channels = number of outgoing filters
                filters = trainable([fh, fw, fic, foc], 'filters')
                
                # stride shape
                sN = 1 # batch = 1 (why anything but 1 ?)
                sH = 1 # height of stride = 1 for 1D conv
                sW = 1 # width of stride = downsampling factor = 1 for no downsampling or > 1 for downsampling
                sC = 1 # depth = number of channels the stride walks over = 1 (why anything but 1 ?)
                strides_no_downsample = [sN, sH, sW, sC]
                layer = tf.nn.conv2d(layer, filters, strides_no_downsample, 'SAME')
                
                # shape of biases = [num outgoing filters]
                biases = trainable([foc], 'biases')
                layer = tf.nn.bias_add(layer, biases)
                layer = tf.nn.leaky_relu(layer)
                #layer = tf.nn.dropout(layer, self.downsample_keep_prob)
                layer = tf.contrib.layers.batch_norm(layer, is_training = self.is_training)
                
                # downsample
                with tf.variable_scope('downsample'):
                    fw = 5#1 + fw*2
                    filters = trainable([fh, fw, foc, foc], 'filters')

                    sW = 2
                    strides_no_downsample = [sN, sH, sW, sC]
                    layer = tf.nn.conv2d(layer, filters, strides_no_downsample, 'SAME')

                    # shape of biases = [num outgoing filters]
                    biases = trainable([foc], 'biases')
                    layer = tf.nn.bias_add(layer, biases)
                    layer = tf.nn.leaky_relu(layer)
                    layer = tf.nn.dropout(layer, self.downsample_keep_prob)
                    layer = tf.contrib.layers.batch_norm(layer, is_training = self.is_training)

                self._layers.append(layer)
                              
        
                
        self._msg += ' '
        with tf.variable_scope('labels'):
            self._msg += '.'; self._vprint(self._msg)
            
            self.labels = tf.placeholder(dtype = self.dtype, shape = [None, self.num_classes], name = 'labels')
            self.true_cls = tf.argmax(self.labels, axis = 1)
            
        with tf.variable_scope('logits'):
            self._msg += '.'; self._vprint(self._msg)
            
            self._logits = tf.contrib.layers.fully_connected(tf.layers.flatten(self._layers[-1]), self.num_classes, activation_fn = None)

        with tf.variable_scope('predictions'):
            self._msg += '.'; self._vprint(self._msg)
            
            self.predictions = tf.nn.softmax(self._logits)
            
            self.pred_cls = tf.argmax(self.predictions, axis = 1)

            self.correct_prediction = tf.equal(self.pred_cls, self.true_cls)
            
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, self.dtype))
        
        
        with tf.variable_scope('costs'):
            self._msg += '.'; self._vprint(self._msg)

            self._cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.labels, logits = self._logits)
 
            self.cost = tf.reduce_mean(self._cross_entropy)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope('train'):
                
                self._msg += '.'; self._vprint(self._msg)
                self.optimizer = tf.train.AdamOptimizer(self.adam_initial_learning_rate, epsilon=1e-8).minimize(self.cost)
            
        with tf.variable_scope('logging'):
            self._msg += '.'; self._vprint(self._msg)
            with tf.variable_scope('image'):
                
                self.image_buf = tf.placeholder(tf.string, shape=[])
                epoch_image = tf.expand_dims(tf.image.decode_png(self.image_buf, channels=4), 0)
            
            tf.summary.scalar(name = 'cost', tensor = self.cost)
            tf.summary.scalar(name = 'accuracy', tensor = self.accuracy)
            tf.summary.image('confusion_matrix', epoch_image)
            self.summary = tf.summary.merge_all()
            
        self._msg += ' done'
        self._vprint(self._msg)
        self._msg = ''