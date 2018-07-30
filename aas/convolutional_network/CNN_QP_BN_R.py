# CNN_QP_BN_R

import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '../modules'))

from Restoreable_Component import Restoreable_Component
from Quad_Path_Layer import Quad_Path_Layer

import tensorflow as tf
import numpy as np

class CNN_QP_BN_R(Restoreable_Component):
    """CNN: Convolutional neural network.
       QP: Each computational layer is a quad-path layer.
       BN: All non-linearalities have batch-normalization applied.
       R: Regression, this network predicts a single value for each input."""
    
    def __init__(self,
                 name,
                 wide_convolution_filter_widths,
                 layer_downsampling_factors, 
                 num_1x1_conv_filters_per_layer,
                 log_dir = 'logs/',
                 dtype = tf.float32,
                 adam_initial_learning_rate = 0.0001,
                 cost = 'MSE',
                 accuracy_threshold = 0.00625,
                 gaussian_shift_scalar = 1e-5,
                 verbose = True):
    
        Restoreable_Component.__init__(self, name=name, log_dir=log_dir, verbose=verbose)
                
        self.wide_convolution_filter_widths = wide_convolution_filter_widths
        self.layer_downsampling_factors = layer_downsampling_factors
        
        self.dtype = dtype
        self.adam_initial_learning_rate = adam_initial_learning_rate
        self.cost = cost
        self.accuracy_threshold = accuracy_threshold
        self.gaussian_shift_scalar = gaussian_shift_scalar
        self.num_1x1_conv_filters_per_layer = num_1x1_conv_filters_per_layer 
        
        self.num_freq_channels = 1024
        
    def create_graph(self):
        
        self.save_params()
        self._msg = '\rcreating network graph '; self._vprint(self._msg)
        
        tf.reset_default_graph()
        
        self.is_training = tf.placeholder(dtype = tf.bool, name = 'is_training', shape = [])
        
        with tf.variable_scope('keep_probs'):
            self._msg += '.'; self._vprint(self._msg)
            
            self.sample_keep_prob = tf.placeholder(self.dtype, name = 'sample_keep_prob')
            self.conv_keep_prob = tf.placeholder(self.dtype, name = 'conv_keep_prob') 
            self.pred_keep_prob = tf.placeholder(self.dtype, name = 'pred_keep_prob')    

        with tf.variable_scope('samples'):
            self._msg += '.'; self._vprint(self._msg)
            
            self.samples = tf.placeholder(self.dtype, shape = [None, 1, self.num_freq_channels, 1], name = 'samples')
            self.samples = tf.nn.dropout(self.samples, self.sample_keep_prob)

        self._layers = []
        num_layers = len(self.wide_convolution_filter_widths)
        layer_names = ['layer_{}'.format(i) for i in range(num_layers)]

        for i in range(num_layers):
            self._msg += '.'; self._vprint(self._msg)
            
            # previous layer is input for current layer
            layer = self.samples if i == 0 else self._layers[i - 1]
            layer = Quad_Path_Layer(layer,
                                    layer_names[i],
                                    self.wide_convolution_filter_widths[i],
                                    self.layer_downsampling_factors[i],
                                    self.num_1x1_conv_filters_per_layer[i],
                                    self.conv_keep_prob,
                                    self.is_training,
                                    self.dtype)
            layer = layer.process()
            
            self._layers.append(layer)
            
        with tf.variable_scope('prediction'):
            self._msg += '.'; self._vprint(self._msg)
            reshaped_final_layer = tf.contrib.layers.flatten(self._layers[-1])
            reshaped_final_layer = tf.nn.dropout(reshaped_final_layer, self.pred_keep_prob)
            prediction_weight = tf.get_variable(name = 'weight',
                                                shape = [reshaped_final_layer.get_shape()[-1], 1],
                                                dtype = self.dtype,
                                                initializer = tf.contrib.layers.xavier_initializer())
            
            self.predictions = tf.matmul(reshaped_final_layer, prediction_weight)

        with tf.variable_scope('targets'):
            self._msg += '.'; self._vprint(self._msg)
            
            self.targets = tf.placeholder(dtype = self.dtype, shape = [None, 1], name = 'targets')

        with tf.variable_scope('costs'):
            self._msg += '.'; self._vprint(self._msg)

            error = tf.subtract(self.targets, self.predictions, name = 'error')
            squared_error = tf.square(error, name = 'squared_difference')

            with tf.variable_scope('mean_inverse_shifted_gaussian'):
                self._msg += '.'; self._vprint(self._msg)

                normal_dist = tf.contrib.distributions.Normal(0.0, self.accuracy_threshold, name = 'normal_dist')
                gaussian_prob = normal_dist.prob(error, name = 'gaussian_prob')
                shifted_gaussian = tf.add(gaussian_prob, self.gaussian_shift_scalar, name = 'shifted_gaussian')   

                self.MISG = tf.reduce_mean(tf.divide(1.0, shifted_gaussian), name = 'mean_inverse_shifted_gaussian')
                
            with tf.variable_scope('mean_squared_error'):
                self._msg += '.'; self._vprint(self._msg)
                
                self.MSE = tf.reduce_mean(squared_error)


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
            tf.summary.scalar(name = 'PWT', tensor = self.PWT)
            tf.summary.image('prediction_vs_actual', epoch_image)
            self.summary = tf.summary.merge_all()
            
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope('train'):
                self._msg += '.'; self._vprint(self._msg)
            
                if self.cost == 'MSE':
                    cost = self.MSE
                if self.cost == 'MISG':
                    cost = self.MISG
                if self.cost == 'PWT_weighted_MSE':
                    cost = self.MSE * (100. - self.PWT)
                if self.cost == 'PWT_weighted_MISG':
                    cost = self.MISG * (100. - self.PWT)


                self.optimizer = tf.train.AdamOptimizer(self.adam_initial_learning_rate, epsilon=1e-08).minimize(cost)
            
        
        num_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        self._msg = '\rNetwork Ready - {} trainable parameters'.format(num_trainable_params); self._vprint(self._msg)