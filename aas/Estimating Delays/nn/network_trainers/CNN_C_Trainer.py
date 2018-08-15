"""Summary
"""
# CNN_C_Trainer

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../modules'))
from NN_Trainer import NN_Trainer

import tensorflow as tf
from tensorflow.python.client import timeline

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import io
import itertools

np.seterr(divide='ignore', invalid='ignore') # for cm div/zero (handled)

class CNN_C_Trainer(NN_Trainer):
    """CNN_C_Trainer
        - for training classifcation networks
        - see ????.ipynb for discussion
        
    Args:
        - sample_keep_prob - Probability of keeping a value from an input row
        - conv_keep_prob - Probability of keeping an output value from a convolution
        - num_classes - int - Number of classification classes
            - make sure this matches the value from the network
            - why isnt this number just loaded from the network...
        - single_dataset - bool - train on a single fixed dataset

    """
    __doc__ += NN_Trainer.__doc__

    def __init__(self,
                 network,
                 Data_Creator, # Class
                 num_classes,
                 num_epochs = 100,
                 batch_size = 32,
                 log_dir = '../logs/',
                 model_save_interval = 25,
                 pretrained_model_path = None,
                 metric_names = ['costs', 'accuracies'],
                 sample_keep_prob = 0.80,
                 conv_keep_prob = 0.9,
                 verbose = True,
                 single_dataset = False):
        """__init__"""
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
        self.conv_keep_prob = conv_keep_prob
        self.num_classes = num_classes
        self.single_dataset = single_dataset

    def add_data(self,train_info, test_info, gains, num_flatnesses = 10, abs_min_max_delay = 0.040, precision = 0.00025, blur = 0):
        """Adds data to the Trainer.

            Args

                train_info - (tuple : (visibility data, baselines dict)) - The training data
                test_info - (tuple : (visibility data, baselines dict)) - The testing data
                gains - (dict) - The gains for the visibilites
                num_flatnesses - (int) - Number of flatnesses for each epoch
                    - Number of samples is num_flatnesses * 60
                abs_min_max_delay - (float) - The absolute value of the min or max delay for this dataset.

        """
        self._abs_min_max_delay = abs_min_max_delay

        self._train_batcher = self._Data_Creator(num_flatnesses,
                                                 train_info[0],
                                                 train_info[1],
                                                 gains,
                                                 abs_min_max_delay,
                                                 precision,
                                                 single_dataset = self.single_dataset,
                                                 blur = blur)
        self._train_batcher.gen_data()


        self._test_batcher = self._Data_Creator(num_flatnesses,
                                                test_info[0],
                                                test_info[1],
                                                gains,
                                                abs_min_max_delay,
                                                precision,
                                                single_dataset = self.single_dataset,
                                                blur = blur)
        self._test_batcher.gen_data()

    def train(self):
        
        self.save_params()



        costs = []
        accuracies = []

        tf.reset_default_graph()
        
        self._network.create_graph()
        self._network.save_params()
        
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
                    
                    if self.single_dataset == True and epoch == 0:
                        training_inputs, training_labels, training_bl_dict_singleset = self._train_batcher.get_data(); self._train_batcher.gen_data()
                        testing_inputs, testing_labels, testing_bl_dict_singleset = self._test_batcher.get_data(); self._test_batcher.gen_data()
                        np.savez(archive_loc + '/params/train_bl_dict_singleset', training_bl_dict_singleset)   
                        np.savez(archive_loc + '/params/test_bl_dict_singleset', testing_bl_dict_singleset)
                        
                    if self.single_dataset == False:
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


                        training_inputs_batch = training_inputs[j*batch_size:(j + 1)*batch_size].reshape(-1,1,1024,1)
                        training_labels_batch = training_labels[j*batch_size:(j + 1)*batch_size].reshape(-1,self.num_classes)

                        feed_dict = {self._network.X: training_inputs_batch,
                                     self._network.labels: training_labels_batch,
                                     self._network.sample_keep_prob : self.sample_keep_prob,
                                     self._network.conv_keep_prob : self.conv_keep_prob,
                                     self._network.is_training : True}

                        batch_cost, batch_acc, _ = session.run([self._network.cost, self._network.accuracy, self._network.optimizer], feed_dict = feed_dict)
                        if epoch != 0:
                            self._msg += " batch cost: {:0.4f}".format(batch_cost)
                            self._msg += " batch accs: {:2.2f}".format(batch_acc)
                        self._vprint(self._msg); 
                        
                            
                    train_feed_dict = {self._network.X: training_inputs.reshape(-1,1,1024,1),
                                       self._network.labels: training_labels.reshape(-1,self.num_classes),
                                       self._network.sample_keep_prob : 1.,
                                       self._network.conv_keep_prob : 1.,
                                       self._network.is_training : False}

                    train_predicts =  session.run([self._network.predictions], train_feed_dict)

                    train_feed_dict[self._network.image_buf] = self.plt_confusion_matrix(training_labels.reshape(-1,self.num_classes), train_predicts)

                    training_cost, training_acc, training_summary = session.run([self._network.cost,
                                                                                 self._network.accuracy,
                                                                                 self._network.summary],
                                                                                 feed_dict = train_feed_dict) 

                    training_writer.add_summary(training_summary, epoch)
                    training_writer.flush()  
                
                    
                    test_feed_dict = {self._network.X: testing_inputs.reshape(-1,1,1024,1),
                                      self._network.labels: testing_labels.reshape(-1,self.num_classes),
                                      self._network.sample_keep_prob : 1.,
                                      self._network.conv_keep_prob : 1.,
                                      self._network.is_training : False}

                    test_predicts =  session.run([self._network.predictions], test_feed_dict)

                    test_feed_dict[self._network.image_buf] = self.plt_confusion_matrix(testing_labels.reshape(-1,self.num_classes), test_predicts) 

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
        
    def plt_confusion_matrix(self, labels, pred, normalize=False, title='Confusion matrix'):
        """
        Given one-hot encoded labels and preds, displays a confusion matrix.
        Arguments:
            `labels`:
                The ground truth one-hot encoded labels.
            `pred`:
                The one-hot encoded labels predicted by a model.
            `normalize`:
                If True, divides every column of the confusion matrix
                by its sum. This is helpful when, for instance, there are 1000
                'A' labels and 5 'B' labels. Normalizing this set would
                make the color coding more meaningful and informative.
        """
        labels = [label.argmax() for label in np.asarray(labels).reshape(-1,self.num_classes)] # bc
        pred = [label.argmax() for label in np.asarray(pred).reshape(-1,self.num_classes)] #bc

        #classes = ['pos','neg']#np.arange(len(set(labels)))
        if self.num_classes == 9:
            precision = 0.005
        if self.num_classes == 41:
            precision = 0.001
        if self.num_classes == 81:
            precision = 0.0005
        if self.num_classes == 161:
            precision = 0.00025
        if self.num_classes == 401:
            precision = 0.0001
        if self.num_classes == 2:
            classes = ['pos','neg']
            cm = confusion_matrix(labels, pred)
        else:
            classes = np.round(np.arange(0,0.04 + precision, precision), 5)
            cm = confusion_matrix(labels, pred, np.arange(len(classes)))

        #if normalize:
        cm = cm.astype('float')*100 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

        fig, ax = plt.subplots(figsize = (5,5), dpi = 320)
        #plt.figure(figsize=(15,10))
        im = ax.imshow(cm, interpolation='nearest', aspect='auto', cmap='Oranges', vmin = 0, vmax = 100)
        ax.set_title(title)
        cbar = fig.colorbar(im)
        if len(classes) <= 161:
            tick_marks = np.arange(len(classes))

            fs = 4 if self.num_classes <= 41 else 3
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(classes, fontsize=fs, rotation=-90,  ha='center')
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(classes, fontsize=fs)
            ax.xaxis.set_tick_params(width=0.5)
            ax.yaxis.set_tick_params(width=0.5)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        if self.num_classes <= 41:
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                #s = '{:2.0}'.format(cm[i, j]) if cm[i,j] >= 1 else '.'
                ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.',
                        horizontalalignment="center", fontsize=5, verticalalignment='center', color= "black")

        fig.set_tight_layout(True)
        # plt.show()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi = 320)
        plt.close(fig)
        buf.seek(0)

        return buf.getvalue()

