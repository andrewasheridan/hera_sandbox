"""Summary
"""
# CNN_DSFC_BN_R

import sys, os

from RestoreableComponent import RestoreableComponent

import tensorflow as tf
import numpy as np

class CNN_DSFC_BN_R(RestoreableComponent):

    """CNN_DSFC_BN_R
    
    CNN: Convolutional Neural Network.
    DS: DownSampling. Each convolution is followed by a downsampling convolution
    FC: After the chain of convolutions there is a chain of fully connected layers.
    BN: All non-linearalities have batch-normalization applied.
    R: Regression, this network takes in a sample and returns a value.

    Each layer starts with a 1x3 convolution with 2**i filters (i = layer index) with stride of 1.
    This is fed into a 1x5 convolution with the same number of filters, but stride of 2 (50% downsample)

    Leaky_ReLU and batch normalization is applied after each convolution. Downsamples convolutions have dropout.
    Biases are added before activations.

    Output of last layer is fed to fully connected layer with relu activation

    Cost function is mean squared error
    
    Attributes:
        accuracy_threshold (float, optional): Threshold to count a prediction as being on target
        adam_initial_learning_rate (float): Adam optimizer initial learning rate
        cost (str): name of cost function. 'MSE', 'MQE', 'MISG', 'PWT_weighted_MSE', 'PWT_weighted_MISG'
            - use 'MSE', others experimental        downsample_keep_prob (TYPE): Description
        dtype (tensorflow obj): Type used for all computations
        gaussian_shift_scalar (float): value to shift gaussian for 'MISG' cost
        image_buf (tensorflow obj): buffer for image summaries for tensorboard
        is_training (tensorflow obj): flag for batch normalization
        layer_nodes (TYPE): Description
        MISG (tensorflow obj): cost function, experimental. Mean Inverse Shifted Gaussian 
        MQE (tensorflow obj): cost function, experimental. Mean Quad Error
        MSE (tensorflow obj): cost function. Mean Squared Error
        num_downsamples (TYPE): Description
        optimizer (tensorflow obj): optimization function
        predictions (tensorflow obj): predicted outputs
        PWT (tensorflow obj): Percent of predictions within threshold
        sample_keep_prob (tensorflow obj): keep prob for input samples
        summary (tensorflow obj): summary operation for tensorboard
        targets (tensorflow obj): true values for optimizer
        X (tensorflow obj): input samples
    """
    
    def __init__(self,
                 name,
                 num_downsamples,
                 cost = 'MSE',
                 log_dir = '../logs/',
                 dtype = tf.float32,
                 adam_initial_learning_rate = 0.0001,
                 accuracy_threshold = 0.00625,
                 gaussian_shift_scalar = 1e-5,
                 verbose = True):
        """Summary
        
        Args:
            name (str): name of network
            num_downsamples (TYPE): Description
            cost (str): name of cost function. 'MSE', 'MQE', 'MISG', 'PWT_weighted_MSE', 'PWT_weighted_MISG'
                - use 'MSE', others experimental
            log_dir (str, optional): directory to store network model and params
            dtype (tensorflow obj): Type used for all computations
            adam_initial_learning_rate (tensorflow obj): Adam optimizer initial learning rate
            accuracy_threshold (float, optional): Threshold to count a prediction as being on target
            gaussian_shift_scalar (float): value to shift gaussian for 'MISG' cost
            verbose (bool, optional): be verbose
        """
        RestoreableComponent.__init__(self, name=name, log_dir=log_dir, verbose=verbose)
                
        self.num_downsamples = num_downsamples
        
        self.dtype = dtype
        self.adam_initial_learning_rate = adam_initial_learning_rate
        self.accuracy_threshold = accuracy_threshold
        self.gaussian_shift_scalar = gaussian_shift_scalar
        self.cost = cost
        

        self._num_freq_channels = 1024
        self._layers = []
        self._num_outputs = 1 # Regression


    def create_graph(self):
        """create_graph

        Create the network graph for use in a tensorflow session
        """
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
                fw = 3 # filter width
                fic = 2**(i) # num in channels = number of incoming filters
                foc = 2**(i+1) # num out channels = number of outgoing filters
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
                layer = tf.contrib.layers.batch_norm(layer, is_training = self.is_training)
                
                # downsample
                with tf.variable_scope('downsample'):
                    fw = 5
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

        self.layer_nodes = 2**np.arange(0,10)[:self.num_downsamples][::-1]
        for j in range(len(self.layer_nodes)):
            
            self._msg += '.'; self._vprint(self._msg)
            
            with tf.variable_scope('fc_layer_{}'.format(j)):
                layer = tf.contrib.layers.fully_connected(self._layers[-1], self.layer_nodes[j])
                layer = tf.contrib.layers.batch_norm(layer, is_training = self.is_training)
                self._layers.append(layer)
                              
        
                
        self._msg += ' '
        with tf.variable_scope('targets'):
            self._msg += '.'; self._vprint(self._msg)
            
            self.targets = tf.placeholder(dtype = self.dtype, shape = [None, self._num_outputs], name = 'labels')

        with tf.variable_scope('predictions'):
            self._msg += '.'; self._vprint(self._msg)
            
            self.predictions = tf.contrib.layers.fully_connected(tf.layers.flatten(self._layers[-1]), self._num_outputs)
            
        with tf.variable_scope('costs'):
            self._msg += '.'; self._vprint(self._msg)

            error = tf.subtract(self.targets, self.predictions, name = 'error')
            squared_error = tf.square(error, name = 'squared_difference')
            quad_error = tf.square(squared_error, name = 'quad_error' )

            with tf.variable_scope('mean_inverse_shifted_gaussian'):
                self._msg += '.'; self._vprint(self._msg)

                normal_dist = tf.contrib.distributions.Normal(0.0, self.accuracy_threshold, name = 'normal_dist')
                gaussian_prob = normal_dist.prob(error, name = 'gaussian_prob')
                shifted_gaussian = tf.add(gaussian_prob, self.gaussian_shift_scalar, name = 'shifted_gaussian')   

                self.MISG = tf.reduce_mean(tf.divide(1.0, shifted_gaussian), name = 'mean_inverse_shifted_gaussian')
                
            with tf.variable_scope('mean_squared_error'):
                self._msg += '.'; self._vprint(self._msg)
                
                self.MSE = tf.reduce_mean(squared_error / self.gaussian_shift_scalar)
                
            with tf.variable_scope('mean_quad_error'):
                self._msg += '.'; self._vprint(self._msg)
                
                self.MQE = tf.reduce_mean(quad_error / self.gaussian_shift_scalar)

        with tf.variable_scope('logging'):  

            with tf.variable_scope('image'):
                self._msg += '.'; self._vprint(self._msg)
                
                self.image_buf = tf.placeholder(tf.string, shape=[])
                epoch_image = tf.expand_dims(tf.image.decode_png(self.image_buf, channels=4), 0)

            with tf.variable_scope('percent_within_threshold'):
                self._msg += '.'; self._vprint(self._msg)
                
                self.PWT = 100.*tf.reduce_mean(tf.cast(tf.less_equal(tf.abs(self.targets - self.predictions), self.accuracy_threshold), self.dtype) )


            tf.summary.histogram(name = 'targets', values = self.targets)
            tf.summary.histogram(name = 'predictions',values =  self.predictions)
            tf.summary.scalar(name = 'MSE', tensor = self.MSE)
            tf.summary.scalar(name = 'MISG', tensor = self.MISG)
            tf.summary.scalar(name = 'MQE', tensor = self.MQE)
            tf.summary.scalar(name = 'PWT', tensor = self.PWT)
            tf.summary.image('prediction_vs_actual', epoch_image)
            self.summary = tf.summary.merge_all()
            
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope('train'):
                self._msg += '.'; self._vprint(self._msg)
            
                if self.cost == 'MSE':
                    cost = self.MSE
                if self.cost == 'MQE':
                    cost = tf.log(self.MQE)
                if self.cost == 'MISG':
                    cost = self.MISG
                if self.cost == 'PWT_weighted_MSE':
                    cost = self.MSE * (100. - self.PWT)
                if self.cost == 'PWT_weighted_MISG':
                    cost = self.MISG * (100. - self.PWT)


                self.optimizer = tf.train.AdamOptimizer(self.adam_initial_learning_rate, epsilon=1e-08).minimize(cost)
            
        
        num_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        self._msg = '\rNetwork Ready - {} trainable parameters'.format(num_trainable_params); self._vprint(self._msg)