"""FNN_BN_R
"""

import sys, os

from RestoreableComponent import RestoreableComponent

import tensorflow as tf
import numpy as np

class FNN_BN_R(RestoreableComponent):

    """FNN_BN_R

    A neural network of fully connected layers.
    
    Attributes:
        accuracy_threshold (float): the value for a target to be considered 'good'
        adam_initial_learning_rate (float): Adam optimizer initial learning rate
        cost (str): name of cost function. 'MSE', 'MQE', 'MISG', 'PWT_weighted_MSE', 'PWT_weighted_MISG'
            - use 'MSE', others experimental
        dtype (tensorflow obj): type used for all computations
        fcl_keep_prob (tensorflow obj): keep prob for fully connected layers
        gaussian_shift_scalar (float): value to shift gaussian for 'MISG' cost
        image_buf (tensorflow obj): buffer for image summaries for tensorboard
        is_training (tensorflow obj): flag for batch normalization
        layer_nodes (list of ints): numbers of nodes in each layer
        MISG (tensorflow obj): cost function, experimental. Mean Inverse Shifted Gaussian 
        MQE (tensorflow obj): cost function, experimental. Mean Quad Error
        MSE (tensorflow obj): cost function. Mean Squared Error
        optimizer (tensorflow obj): optimization function
        predictions (tensorflow obj): predicted outputs
        PWT (tensorflow obj): Percent of predictions within threshold
        sample_keep_prob (tensorflow obj): keep prob for input samples
        summary (tensorflow obj): summary operation for tensorboard
        targets (tensorflow obj): true values for optimizer
        X (tensorflow obj): input samples
    """
    __doc__ += RestoreableComponent.__doc__
    
    def __init__(self,
                 name,
                 layer_nodes,
                 cost = 'MSE',
                 log_dir = '../logs/',
                 dtype = tf.float32,
                 adam_initial_learning_rate = 0.0001,
                 accuracy_threshold = 0.00625,
                 gaussian_shift_scalar = 1e-5,
                 verbose = True):
        """__init__
        
        Args:
            name (str): name of network
            layer_nodes (list of ints): numbers of nodes in each layer
            cost (str, optional): name of cost function. 'MSE', 'MQE', 'MISG', 'PWT_weighted_MSE', 'PWT_weighted_MISG'
                    - use 'MSE', others experimental
            log_dir (str, optional): Directory to store network model and params
            dtype (tensorflow obj, optional): type used for all computations
            adam_initial_learning_rate (float, optional): Adam optimizer initial learning rate
            accuracy_threshold (float, optional): the value for a target to be considered 'good'
            gaussian_shift_scalar (float, optional): value to shift gaussian for 'MISG' cost
            verbose (bool, optional): Be verbose   
        """
        RestoreableComponent.__init__(self, name=name, log_dir=log_dir, verbose=verbose)
                
        self.layer_nodes = layer_nodes
        
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
            self.fcl_keep_prob = tf.placeholder(self.dtype, name = 'fcl_keep_prob')
        
        
        with tf.variable_scope('sample'):
            
            self._msg += '.'; self._vprint(self._msg)
            
            self.X = tf.placeholder(self.dtype, shape = [None,self._num_freq_channels], name = 'X')
            self.X = tf.nn.dropout(self.X, self.sample_keep_prob)
            
        with tf.variable_scope('input_layer'):
            
            b = tf.get_variable(name = 'biases', shape = [self.layer_nodes[0]],
                                initializer = tf.contrib.layers.xavier_initializer())
            
            w = tf.get_variable(name = 'weights', shape  = [1024, self.layer_nodes[0]],
                                initializer = tf.contrib.layers.xavier_initializer())
            
            layer = tf.nn.leaky_relu(tf.matmul(self.X, w) + b)
            layer = tf.contrib.layers.batch_norm(layer, is_training = self.is_training)
            layer = tf.nn.dropout(layer, self.fcl_keep_prob)
            self._layers.append(layer)
            
        for i in range(len(self.layer_nodes)):
            if i > 0:
                with tf.variable_scope('layer_%d' %(i)):
                    layer = tf.contrib.layers.fully_connected(self._layers[i-1], self.layer_nodes[i])
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