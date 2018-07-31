# CNN_BC_Trainer

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../modules'))
from NN_Trainer import NN_Trainer

import tensorflow as tf
from tensorflow.python.client import timeline

import numpy as np

class CNN_BC_Trainer(NN_Trainer):

    def __init__(self,
                 network,
                 Data_Creator, # Class
                 num_epochs = 100,
                 batch_size = 32,
                 log_dir = 'logs/',
                 model_save_interval = 25,
                 pretrained_model_path = None,
                 metric_names = ['costs', 'accuracies'],
                 sample_keep_prob = 0.80,
                 downsample_keep_prob = 0.9,
                 verbose = True):
    
        NN_Trainer.__init__(self,
                            network = network,
                            Data_Creator = Data_Creator,
                            num_epochs = num_epochs,
                            batch_size = batch_size,
                            log_dir = log_dir,
                            model_save_interval = model_save_interval,
                            pretrained_model_path = pretrained_model_path,
                            metric_names = metric_names,
                            verbose = verbose)
        

        self.sample_keep_prob = sample_keep_prob
        self.downsample_keep_prob = downsample_keep_prob

    def train(self):
        
        self.save_params()

        costs = []
        accuracies = []

        tf.reset_default_graph()
        
        self._network.create_graph()
        saver = tf.train.Saver()

        with tf.Session() as session:
            
            if self.pretrained_model_path == None:
                session.run(tf.global_variables_initializer())
                
            else:
                saver.restore(session, self.pretrained_model_path)

            archive_loc = self.log_dir + self._network.name
            training_writer = tf.summary.FileWriter(archive_loc + '/training', session.graph)
            testing_writer = tf.summary.FileWriter(archive_loc + '/testing', session.graph)
            self.model_save_location = archive_loc + '/trained_model.ckpt'   
            
            
            self._msg = '\rtraining';self._vprint(self._msg)

            try:
                for epoch in range(self.num_epochs):
                    
                    

                    training_inputs, training_labels = self._train_batcher.get_data(); self._train_batcher.gen_data()
                    testing_inputs, testing_labels = self._test_batcher.get_data(); self._test_batcher.gen_data()  

                    training_labels = np.asarray(training_labels)
                    testing_labels = np.asarray(testing_labels)
                    
                    # if the division here has a remainde some values are just truncated
                    batch_size = self.batch_size
                    num_entries = training_inputs.shape[0]

                    for j in range(int(num_entries/batch_size)):
                        self._msg = '\repoch'
                        self._msg += '- {:5.0f}/{:5.0f}'.format(epoch + 1,self.num_epochs)
                        self._msg += ' - batch: {:4.0f}/{:4.0f}'.format(j + 1, int(num_entries/batch_size))
                        if epoch != 0:
                            self._msg += ' - (Training, Testing) - '.format(epoch)
                            self._msg += " costs: ({:0.4f}, {:0.4f})".format(training_cost, testing_cost)
                            self._msg += " accss: ({:2.2f}, {:2.2f})".format(training_acc, testing_acc)
                        self._vprint(self._msg); 

                        training_inputs_batch = training_inputs[j*batch_size:(j + 1)*batch_size].reshape(-1,1,1024,1)
                        training_labels_batch = training_labels[j*batch_size:(j + 1)*batch_size].reshape(-1,2)

                        feed_dict = {self._network.X: training_inputs_batch,
                                     self._network.labels: training_labels_batch,
                                     self._network.sample_keep_prob : self.sample_keep_prob,
                                     self._network.downsample_keep_prob : self.downsample_keep_prob,
                                     self._network.is_training : True}

                        session.run([self._network.optimizer], feed_dict = feed_dict) 
                            
                    train_feed_dict = {self._network.X: training_inputs.reshape(-1,1,1024,1),
                                       self._network.labels: training_labels.reshape(-1,2),
                                       self._network.sample_keep_prob : 1.,
                                       self._network.downsample_keep_prob : 1.,
                                       self._network.is_training : False}


                    training_cost, training_acc, training_summary = session.run([self._network.cost,
                                                                                 self._network.accuracy,
                                                                                 self._network.summary],
                                                                                 feed_dict = train_feed_dict) 

                    training_writer.add_summary(training_summary, epoch)
                    training_writer.flush()  
                
                    
                    test_feed_dict = {self._network.X: testing_inputs.reshape(-1,1,1024,1),
                                      self._network.labels: testing_labels.reshape(-1,2),
                                      self._network.sample_keep_prob : 1.,
                                      self._network.downsample_keep_prob : 1.,
                                      self._network.is_training : False} 

                    testing_cost, testing_acc, testing_summary = session.run([self._network.cost,
                                                                              self._network.accuracy,
                                                                              self._network.summary],
                                                                              feed_dict = test_feed_dict)

                    testing_writer.add_summary(testing_summary, epoch)
                    testing_writer.flush()
                    
                    costs.append((training_cost, testing_cost))
                    accuracies.append((training_acc, testing_acc))
                    
                    if (epoch + 1) % self.model_save_interval == 0:
                        saver.save(session, self.model_save_location, epoch + 1)
                
                self.msg = ''
            except KeyboardInterrupt:
                self._msg = ' TRAINING INTERUPPTED' # this never prints I dont know why
                pass

            self._msg += '\rtraining ended'; self._vprint(self._msg)
            
            training_writer.close()
            testing_writer.close()


        session.close()
        self._msg += ' - session closed'; self._vprint(self._msg)
        self._msg = ''

        self._metrics = [costs, accuracies]
        self.save_metrics()
