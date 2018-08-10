# Estdel_NN

import sys
import numpy as np
import tensorflow as tf

class Estdel_NN(object):
    """ Estdel_NN - Copy of CNN_DS_BN_C (Aug_7)

        CNN: Convolutional Neural Network.
        DS: DownSampling. Each layer ends with a downsampling convolution
        BN: All non-linearalities have batch-normalization applied.
        C: Classification, this network classifies samples as having one of many labels.

        Network Structure:

            Incoming sample has dropout sample_keep_prob - set on evaluation
 
            Each layer has 4 convolutions, each with 4**(i+1) filters (i = zero based layer index).
                - Each convolution:
                    - is a 1D convolution
                    - feeds into the next
                    - has a filter of size (1, fw)
                    - has biases and a LeakyReLU activation
                    - has dropout with conv_keep_prob - set on evaluation
                    - has batch normalization
                - Filter widths (fw) of the four convolutions are [3,5,7,9]
                TODO: Optional filter widths? (different per layer?)
                - Fourth convolution will do 50% downsample (horizontal stride = 2)
                
            Final convolution feeds into fully connected layer as logits
            Cost function softmax_cross_entropy_with_logits_v2
            Optimizer is Adam with adam_initial_learning_rate
            Predictions are softmax probabilites
        
        Args:
            num_downsamples (int)   - The number of downsample convolutions 
                                    - The number of layers
            num_classes (int)   - The number of classification labels
                                - Must be one of : 9, 81, 161, 401
            dtype (class)   - Datatype for network. tf.float16 will change Adam epsilon from 1e-8 to 1e-4
            adam_initial_learning_rate (float)  -   Adam optimizer initial learning rate
                                                -   Will decay over time as per Adam docs.
            verbose (bool)  - Be verbose.

        Usage :
            Evaluation:
                - run create_graph()
                - start tensorflow session

        Methods:
            - create_graph() - Contructs the network graph

    """

    def __init__(self,
                 num_downsamples,
                 num_classes,
                 dtype = tf.float32,
                 adam_initial_learning_rate = 0.0001,
                 verbose = True):
    
                
        self.num_downsamples = num_downsamples
        self.dtype = dtype
        self.adam_initial_learning_rate = adam_initial_learning_rate
        self._num_freq_channels = 1024
        self._layers = []
        self.num_classes = num_classes #  classifier...
        self._vprint = sys.stdout.write if verbose else lambda *a, **k: None # for verbose printing


    def create_graph(self):
        """ Create the network graph for use in a tensorflow session"""

        
        self._msg = '\rcreating network graph '; self._vprint(self._msg)

        
        tf.reset_default_graph()
        self.is_training = tf.placeholder(dtype = tf.bool, shape = [], name = 'is_training')
        
        with tf.variable_scope('keep_probs'):
            self._msg += '.'; self._vprint(self._msg)

            self.sample_keep_prob = tf.placeholder(self.dtype, name = 'sample_keep_prob')
            self.conv_keep_prob = tf.placeholder(self.dtype, name = 'conv_keep_prob')
        
        with tf.variable_scope('sample'):
            self._msg += '.'; self._vprint(self._msg)
            
            self.X = tf.placeholder(self.dtype, shape = [None, 1, self._num_freq_channels, 1], name = 'X')
            self.X = tf.nn.dropout(self.X, self.sample_keep_prob)

        for i in range(self.num_downsamples):
            self._msg += '.'; self._vprint(self._msg)
            
            # foc - filter out channels - number of filters
            foc = 4**(i+1) # num filters grows with each downsample
            
            with tf.variable_scope('downsample_{}'.format(i)):

                layer = self.X if i == 0 else self._layers[-1]
                fitlter_widths = [3,5,7,9]

                for fw in fitlter_widths:
                    with tf.variable_scope('conv_1x{}'.format(fw)):
                        
                        layer = tf.layers.conv2d(inputs = layer,
                                                 filters = foc, 
                                                 kernel_size = (1,fw),
                                                 strides = (1,1) if fw != fitlter_widths[-1] else (1,2), # downscale last conv
                                                 padding = 'SAME',
                                                 activation = tf.nn.leaky_relu,
                                                 use_bias = True,
                                                 bias_initializer = tf.contrib.layers.xavier_initializer())
                        
                        layer = tf.nn.dropout(layer, self.conv_keep_prob)
                        layer = tf.layers.batch_normalization(layer, training = self.is_training)
                        
            self._layers.append(layer)              
                
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
        
        # batch normalization requires the update_ops variable and control dependency
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope('train'):
                self._msg += '.'; self._vprint(self._msg)

                if self.dtype == tf.float16:
                    epsilon=1e-4 # optimizer outputs NaN otherwise :(
                else:
                    epsilon=1e-8

                self.optimizer = tf.train.AdamOptimizer(self.adam_initial_learning_rate, epsilon=epsilon).minimize(self.cost)
            
        with tf.variable_scope('logging'):
            self._msg += '.'; self._vprint(self._msg)

            with tf.variable_scope('image'):
                
                self.image_buf = tf.placeholder(tf.string, shape=[])
                epoch_image = tf.expand_dims(tf.image.decode_png(self.image_buf, channels=4), 0)
            
            tf.summary.scalar(name = 'cost', tensor = self.cost)
            tf.summary.scalar(name = 'accuracy', tensor = self.accuracy)
            tf.summary.image('confusion_matrix', epoch_image)

            self.summary = tf.summary.merge_all()
            
        num_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        self._msg = '\rnetwork Ready - {} parameters'.format(num_trainable_params); self._vprint(self._msg)
