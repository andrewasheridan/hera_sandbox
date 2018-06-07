import tensorflow as tf
import numpy as np
import sys
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '/Users/andrew/Documents/Lab/aas/Hera/modules'))
from NN_helpers import gen_plot

class Sheridan_NN_One(object):
    """A neural network of fully connected ReLU layers. First layer is LeakyReLU followed by dropout."""
    
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
            
        # hidden layers
        for i in range(len(layer_nodes)):
            if i > 0:
                with tf.variable_scope('layer_%d' %i):
                    b = tf.get_variable(name = 'biases_%d' %i,
                                        shape = [layer_nodes[i]],
                                        initializer = tf.zeros_initializer())
                    
                    w = tf.get_variable(name = 'weights_%d' %i,
                                        shape  = [layer_nodes[i-1], layer_nodes[i]],
                                        initializer = tf.contrib.layers.xavier_initializer())
                    
                    layer = tf.nn.leaky_relu(tf.matmul(layers[i], w) + b)
                    layers.append(layer)
                    
        with tf.variable_scope('output_layer'):
            b = tf.get_variable(name = "biases_out",
                                shape = [number_of_outputs],
                                initializer = tf.zeros_initializer())
            
            w = tf.get_variable(name = "weights_out",
                                shape  = [layer_nodes[-1], number_of_outputs],
                                initializer = tf.contrib.layers.xavier_initializer())
            
            self.prediction = tf.nn.relu(tf.matmul(layers[-1], w) + b)
            
        with tf.variable_scope('cost'):
            self.cost = tf.reduce_mean(tf.squared_difference(self.prediction, self.y))

        with tf.variable_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-08).minimize(self.cost)

        with tf.variable_scope('image'):
            self.image_buf = tf.placeholder(tf.string, shape=[])
            epoch_image = tf.expand_dims(tf.image.decode_png(self.image_buf, channels=4), 0)
    
        with tf.variable_scope('logging'):
            tf.summary.image('prediction_vs_actual', epoch_image)
            tf.summary.histogram('predictions', self.prediction)
            tf.summary.scalar('current_cost', self.cost)
            self.summary = tf.summary.merge_all()
            
            
def sheridan_train(X_train, y_train,
                   X_test,  y_test,
                   num_epochs, num_batches,
                   network, output_scaler, log_direc,
                   keep_prob_rate = 0.50,model_save_interval = 50):
    
    """For training a NN. Currently only tested with network = Sheridan_NN_One()."""
    
    number_of_inputs  = X_train.shape[1]
    number_of_outputs = y_train.shape[1]

    saver = tf.train.Saver()
    
    with tf.Session() as session:

        # reset variables
        session.run(tf.global_variables_initializer())
        
        # setup logging
        training_writer = tf.summary.FileWriter(log_direc + '/training', session.graph)
        testing_writer = tf.summary.FileWriter(log_direc + '/testing', session.graph)
        model_save_location = log_direc + '/trained_model.ckpt'    

        
        # train
        for epoch in range(num_epochs):

            # break training data into batches
            split_permutation = np.random.permutation(X_train.shape[0])
            X_train_batches = np.vsplit(X_train[split_permutation], num_batches)
            y_train_batches = np.vsplit(y_train[split_permutation], num_batches)
            
            # train on batches
            for i in range(num_batches):
                session.run(network.optimizer,
                            feed_dict = {network.X: X_train_batches[i], 
                                         network.y: y_train_batches[i], 
                                         network.keep_prob : keep_prob_rate})
                sys.stdout.write('\repoch: {:4.0f} -- testing_cost: --.--------- -- batch: {:4.0f}'.format(epoch, i))
                
            # training summaries
            prediction_scaled_test = session.run(network.prediction,
                                                 feed_dict = {network.X: X_train,
                                                              network.keep_prob : 1.00})
            
            training_cost, training_summary = session.run([network.cost, network.summary],
                                                          feed_dict = {network.X: X_train,
                                                                       network.y: y_train,
                                                                       network.keep_prob : 1.00,
                                                                       network.image_buf: gen_plot(prediction_scaled_test, y_train, output_scaler)})
            training_writer.add_summary(training_summary, epoch)
            testing_writer.flush()  
        
             # testing summaries
            prediction_scaled_test = session.run(network.prediction,
                                                 feed_dict = {network.X: X_test,
                                                              network.keep_prob : 1.00})
            
            testing_cost, testing_summary = session.run([network.cost, network.summary],
                                                        feed_dict = {network.X: X_test,
                                                                     network.y: y_test,
                                                                     network.keep_prob : 1.00,
                                                                     network.image_buf: gen_plot(prediction_scaled_test, y_test, output_scaler)})
            testing_writer.add_summary(testing_summary, epoch)
            training_writer.flush()

            # message
            sys.stdout.write('\repoch: {:4.0f} -- testing_cost: {:2.10f} -- batch:'.format(epoch, testing_cost))

            # save model
            if (epoch + 1) % model_save_interval == 0:
                saver.save(session, model_save_location, epoch)
            
        # save last model
        saver.save(session, model_save_location, epoch)      