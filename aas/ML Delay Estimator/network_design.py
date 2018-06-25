import tensorflow as tf

class delay_estimator_network(object):
    
    def __init__(self, layer_nodes, number_of_inputs, number_of_outputs, learning_rate):
        
        tf.reset_default_graph()
        
        with tf.variable_scope('input_X'):
            self.X  = tf.placeholder(tf.float32, shape = (None, number_of_inputs))
    
        with tf.variable_scope('input_y'):
            self.y  = tf.placeholder(tf.float32, shape = (None, 1))
            
        with tf.variable_scope('keep'):
            self.keep_prob = tf.placeholder(tf.float32)
            
        layers = []
        with tf.variable_scope('input_layer'):
            b = tf.get_variable(name = 'biases_input',
                                shape = [layer_nodes[0]],
                                initializer = tf.zeros_initializer())
            
            w = tf.get_variable(name = 'weights_input',
                                shape  = [number_of_inputs, layer_nodes[0]],
                                initializer = tf.contrib.layers.xavier_initializer())
            
            layer = tf.nn.leaky_relu(tf.matmul(self.X, w) + b)
            layers.append(layer)
        
        with tf.variable_scope('dropout'):
            dropout = tf.nn.dropout(layers[0], self.keep_prob)            
            layers.append(dropout)
            
        for i in range(len(layer_nodes)):
            if i > 0:
                with tf.variable_scope('layer_%d' %i):
                    b = tf.get_variable(name = 'biases_%d' %i,
                                        shape = [layer_nodes[i]],
                                        initializer = tf.zeros_initializer())
                    
                    w = tf.get_variable(name = 'weights_%d' %i,
                                        shape  = [layer_nodes[i-1], layer_nodes[i]],
                                        initializer = tf.contrib.layers.xavier_initializer())
                    
                    layer = tf.nn.relu6(tf.matmul(layers[i], w) + b)
                    layers.append(layer)
                    
        with tf.variable_scope('output_layer'):
            b = tf.get_variable(name = "biases_out",
                                shape = [number_of_outputs],
                                initializer = tf.zeros_initializer())
            
            w = tf.get_variable(name = "weights_out",
                                shape  = [layer_nodes[-1], number_of_outputs],
                                initializer = tf.contrib.layers.xavier_initializer())
            
            self.prediction = tf.nn.relu6(tf.matmul(layers[-1], w) + b)





        # For Training
        with tf.variable_scope('cost'):

            self.cost = tf.reduce_mean(tf.squared_difference(self.prediction, self.y))
            
        with tf.variable_scope('acc_test'):
            self.acc_test = tf.placeholder(tf.float32)
            
        with tf.variable_scope('acc_2'):
            sigma = 0.00625
            dist = tf.contrib.distributions.Normal(0., sigma)
            self.acc = tf.reduce_mean(tf.divide(1.,1. + dist.prob(self.prediction - self.y)))
        
        with tf.variable_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-08).minimize(self.acc)

            # initial training was by minizing the cost function
            # switched to acc after a few hundred (?) epochs
            # self.optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-08).minimize(self.cost)


        with tf.variable_scope('image'):
            self.image_buf = tf.placeholder(tf.string, shape=[])
            epoch_image = tf.expand_dims(tf.image.decode_png(self.image_buf, channels=4), 0)
            
        with tf.variable_scope('logging'):                                                                                
            
            tf.summary.image('prediction_vs_actual', epoch_image)
            tf.summary.histogram('predictions', self.prediction)
            tf.summary.scalar('current_cost', self.cost)
            tf.summary.scalar('acc', self.acc)
            tf.summary.scalar('accuracy_0005', self.acc_test)
            self.summary = tf.summary.merge_all()