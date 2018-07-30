# Quad_Path_Layer

import numpy as np

import tensorflow as tf

class Quad_Path_Layer(object):
    
    def __init__(self,
                 t,
                 layer_name,
                 wide_convolution_filter_width,
                 layer_downsampling_factor,
                 num_1x1_conv_filters = 4,
                 conv_keep_prob = 0.90,
                 is_training = True,
                 dtype = tf.float32):
        
        self.t = t
        self.layer_name = layer_name
        self.wide_convolution_filter_width = wide_convolution_filter_width
        self.strides = [1,1,layer_downsampling_factor, 1]
        self.num_1x1_conv_filters = num_1x1_conv_filters
        self.conv_keep_prob = conv_keep_prob
        self.is_training = is_training
        self.dtype = dtype

        
    def process(self):

        with tf.variable_scope(self.layer_name):

            narrow_convolution_filter_width = self.wide_convolution_filter_width / 2

            num_narrow_conv_filters = np.max([self.num_1x1_conv_filters / 2, 2])
            num_wide_conv_filters = np.max([num_narrow_conv_filters / 2, 1])


            path_A = self._avg_scope()
            path_B = self._max_scope()

            path_C_and_D = self._1x1_conv(self.t)

            path_C = self._conv_scope(path_C_and_D,
                                      [1,
                                       narrow_convolution_filter_width,
                                       path_C_and_D.get_shape().as_list()[3],
                                       num_narrow_conv_filters],
                                      self.strides,
                                      scope_name = 'narrow')

            path_D = self._conv_scope(path_C_and_D,
                                      [1,
                                       self.wide_convolution_filter_width,
                                       path_C_and_D.get_shape().as_list()[3],
                                       num_wide_conv_filters],
                                      self.strides,
                                      scope_name = 'wide')

            t = self._filter_cat_scope([path_A, path_B, path_C, path_D])

        return t

    def _trainable(self, name, shape):
        return tf.get_variable(name = name,
                               dtype = self.dtype,
                               shape = shape,
                               initializer = tf.contrib.layers.xavier_initializer())
    
    def _bias_add_scope(self, t, shape):
        """Creates a scope around a trainable bias and its addition to input"""
        with tf.variable_scope('add_bias'):

            biases = self._trainable('biases', shape)
            t = tf.nn.bias_add(t, biases)

        return t


    def _conv_scope(self,t, filter_shape, strides, scope_name = 'convolution'):
        """Creates a scope around a convolution."""
        with tf.variable_scope(scope_name):

            t = tf.nn.conv2d(t, self._trainable('filters', filter_shape),strides,'SAME')
            t = self._bias_add_scope(t, [filter_shape[-1]])
            t = tf.nn.relu(t)
            t = tf.contrib.layers.batch_norm(t, is_training = self.is_training)
            t = tf.nn.dropout(t, self.conv_keep_prob)

        return t
    
    def _1x1_conv(self, t):
        return self._conv_scope(t,
                                [1,1,t.get_shape().as_list()[3],self.num_1x1_conv_filters],
                                [1,1,1,1],
                                "1x1_conv")

    def _avg_scope(self):
        """Creates a scope around the average-pool path."""
        with tf.variable_scope('average'):
            t = tf.nn.avg_pool(self.t, self.strides, self.strides, padding = "SAME")
            t = self._1x1_conv(t)

        return t

    def _max_scope(self):
        """Creates a scope around the max-pool path"""
        with tf.variable_scope('max'):
            t = tf.nn.max_pool(self.t, self.strides, self.strides, padding = "SAME")
            t = self._1x1_conv(t)

        return t

    def _filter_cat_scope(self,t_list):
        """Creates a scope around filter concatation (layer output)"""
        with tf.variable_scope('filter_cat'):
            t = tf.concat(t_list, 3)
        return t