# FNN_BN_R_Trainer

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../modules'))
from NN_Trainer import NN_Trainer

import tensorflow as tf
from tensorflow.python.client import timeline

import numpy as np

class FNN_BN_R_Trainer(NN_Trainer):
    
    def __init__(self,
                 network,
                 Data_Creator, # Class
                 num_epochs = 100,
                 batch_size = 32,
                 log_dir = '../logs/',
                 model_save_interval = 25,
                 pretrained_model_path = None,
                 metric_names = ['MISGs', 'MSEs', 'MQEs', 'PWTs'],
                 sample_keep_prob = 0.80,
                 fcl_keep_prob = 0.9,
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
        self.fcl_keep_prob = fcl_keep_prob
        
    def train(self):
        
        self.save_params()
        
        

        MISGs = []
        MSEs = []
        MQEs = []
        PWTs = []
        
        tf.reset_default_graph()
        
        self._network.create_graph()
        saver = tf.train.Saver()
        
        itx =  lambda x: np.array(x) * 2. * self._abs_min_max_delay - self._abs_min_max_delay
        
        
        with tf.Session() as session:

            if self.pretrained_model_path == None:
                session.run(tf.global_variables_initializer())
                
            else:
                saver.restore(session, self.pretrained_model_path)
                
            options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            archive_loc = self.log_dir + self._network.name 
            training_writer = tf.summary.FileWriter(archive_loc + '/training', session.graph)
            testing_writer = tf.summary.FileWriter(archive_loc + '/testing', session.graph)
            self._model_save_location = archive_loc + '/trained_model.ckpt'   
            
            try:
                for epoch in range(self.num_epochs):


                    training_inputs, training_targets = self._train_batcher.get_data(); self._train_batcher.gen_data()
                    testing_inputs, testing_targets = self._test_batcher.get_data(); self._test_batcher.gen_data()  

                    # if the division here has a remainde some values are just truncated
                    batch_size = self.batch_size
                    num_entries = training_inputs.shape[0]

                    for j in range(int(num_entries/batch_size)):
                        self._msg = '\repoch'
                        self._msg += '- {:5.0f}/{:5.0f}'.format(epoch + 1,self.num_epochs)
                        self._msg += ' - batch: {:4.0f}/{:4.0f}'.format(j + 1, int(num_entries/batch_size))
                        if epoch != 0:
                            self._msg += ' - (Training, Testing) - '.format(epoch)
                            self._msg += ' MISG: ({:0.4f}, {:0.4f})'.format(training_MISG, testing_MISG)
                            self._msg += ' MSE: ({:0.4f}, {:0.4f})'.format(training_MSE, testing_MSE)
                            self._msg += ' MQE: ({:0.4f}, {:0.4f})'.format(training_MQE, testing_MQE)                   
                            self._msg += ' PWT: ({:2.2f}, {:2.2f})'.format(training_PWT, testing_PWT)



                        training_inputs_batch = training_inputs[j*batch_size:(j + 1)*batch_size].reshape(-1,1024)
                        training_targets_batch = training_targets[j*batch_size:(j + 1)*batch_size].reshape(-1,1)

                        feed_dict = {self._network.X: training_inputs_batch,
                                     self._network.targets: training_targets_batch,
                                     self._network.sample_keep_prob : self.sample_keep_prob,
                                     self._network.fcl_keep_prob : self.fcl_keep_prob,
                                     self._network.is_training : True}



                        if j == 0 and (epoch + 1) % self.model_save_interval == 0:

                            session.run([self._network.optimizer], feed_dict = feed_dict,
                                           options = options, run_metadata = run_metadata) 

                            training_writer.add_run_metadata(run_metadata, 'epoch%d' %epoch)

                            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                            chrome_trace = fetched_timeline.generate_chrome_trace_format()
                            direc = archive_loc + '/timelines/'

                            if not os.path.exists(direc):
                                os.makedirs(direc)
                            with open(direc + 'timeline_{}.json'.format(epoch), 'w') as f:
                                f.write(chrome_trace)
                                self._msg += ' saving timeline & metadata'
                        else:
                            session.run([self._network.optimizer], feed_dict = feed_dict) 

                        self._vprint(self._msg)

                    # Prediction: Scaled Train(ing results)   
                    PST = session.run(self._network.predictions,
                                      feed_dict = {self._network.X: training_inputs.reshape(-1,1024),
                                                   self._network.sample_keep_prob : 1.,
                                                   self._network.fcl_keep_prob : 1.,
                                                   self._network.is_training : False}) 

                    train_feed_dict = {self._network.X: training_inputs.reshape(-1,1024),
                                       self._network.targets: training_targets.reshape(-1,1),
                                       self._network.sample_keep_prob : 1.,
                                       self._network.fcl_keep_prob : 1.,
                                       self._network.image_buf: self.gen_plot(PST, training_targets, itx),
                                       self._network.is_training : False}


                    training_MISG, training_MSE, training_MQE, training_PWT, training_summary = session.run([self._network.MISG,
                                                                                                             self._network.MSE,
                                                                                                             self._network.MQE,
                                                                                                             self._network.PWT,
                                                                                                             self._network.summary],
                                                                                                             feed_dict = train_feed_dict) 

                    training_writer.add_summary(training_summary, epoch)
                    training_writer.flush()  

                    # Prediction: Scaled test(ing results)   
                    PST = session.run(self._network.predictions,
                                      feed_dict = {self._network.X: testing_inputs.reshape(-1,1024),
                                                   self._network.sample_keep_prob : 1.,
                                                   self._network.fcl_keep_prob : 1.,
                                                   self._network.is_training : False}) 

                    test_feed_dict = {self._network.X: testing_inputs.reshape(-1,1024),
                                      self._network.targets: testing_targets.reshape(-1,1),
                                      self._network.sample_keep_prob : 1.,
                                      self._network.fcl_keep_prob : 1.,
                                      self._network.image_buf: self.gen_plot(PST,testing_targets, itx),
                                      self._network.is_training : False} 

                    testing_MISG, testing_MSE, testing_MQE, testing_PWT, testing_summary = session.run([self._network.MISG,
                                                                                                        self._network.MSE,
                                                                                                        self._network.MQE,
                                                                                                        self._network.PWT,
                                                                                                        self._network.summary],
                                                                                                        feed_dict = test_feed_dict)




                    testing_writer.add_summary(testing_summary, epoch)
                    testing_writer.flush()  

                    MISGs.append((training_MISG, testing_MISG))
                    MSEs.append((training_MSE, testing_MSE))
                    MQEs.append((training_MQE, testing_MQE))
                    PWTs.append((training_PWT, testing_PWT))

                    if (epoch + 1) % self.model_save_interval == 0:
                        saver.save(session, self._model_save_location, epoch)

            except KeyboardInterrupt:
                
                pass

            self._msg += '\rtraining ended'; self._vprint(self._msg)

            training_writer.close()
            testing_writer.close()

        session.close()
        self._msg += ' - session closed'; self._vprint(self._msg)
        
        self._metrics = [MISGs, MSEs, MQEs, PWTs]
        self.save_metrics()
        