import sys, os
from nn.networks import RestoreableComponent


# NN_Trainer
import matplotlib.pyplot as plt
import numpy as np
import os, io

class NN_Trainer(RestoreableComponent):
    """ Parent for restoreable neural network trainer (does not include training method).
        Add train method particular to network.

        Args:

            network - (one of the network classes from this repo) - The network to train
            Data_Creator - (one of the Data_Creator_* classes from this repo) - Creates data..
            num_epochs - (int) - Number of training epochs
            batch_size - (int) - Number of samples in each batch
                - Number of batches = int(number_of_samples/batch_size)
            log_dir - (string) - Directory to store logs & parameters
            model_save_interval - (int) - Save model every (this many) epochs
            pretrained_model_path - (string) - Path to previously trained model
            metric_names - (list of strings) - Names of the various metrics for this trainer
            verbose - (bool) - Be verbose.
    """
    __doc__ += RestoreableComponent.__doc__
    
    def __init__(self,
                 network,
                 Data_Creator, # Class
                 num_epochs,
                 batch_size,
                 log_dir,
                 model_save_interval,
                 pretrained_model_path,
                 metric_names,
                 verbose = True):
    
        RestoreableComponent.__init__(self, name=network.name, log_dir=log_dir, verbose=verbose)
        
        self._network = network   
        self._Data_Creator = Data_Creator
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model_save_interval = model_save_interval 
        self.pretrained_model_path = pretrained_model_path
        self._metrics = [] # list of lists
        self.metric_names = metric_names # list of strings, one per metric

    def add_data(self,train_info, test_info, gains, num_flatnesses = 10, abs_min_max_delay = 0.040):
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
                                                 abs_min_max_delay)
        self._train_batcher.gen_data()


        self._test_batcher = self._Data_Creator(num_flatnesses,
                                                test_info[0],
                                                test_info[1],
                                                gains,
                                                abs_min_max_delay)
        self._test_batcher.gen_data()
        
    def save_metrics(self):
        """Saves the recorded metrics to disk"""

        self._msg = '\rsaving metrics'; self._vprint(self._msg)
        direc = self.log_dir + self._network.name + '/'

        if not os.path.exists(direc): # should already exist by this point buuuuuut
            self._msg += ' - creating new directory ';self._vprint(self._msg)
            
            os.makedirs(direc)
        np.savez(direc + 'metrics', self.get_metrics()) 
        
        self._msg += ' - saved';self._vprint(self._msg)
        
        
    def get_metrics(self):
        """Returns the recorded metrics (list of lists)"""
        return {self.metric_names[i] : self._metrics[i] for i in range(len(self._metrics))}

    def plot_metrics(self,figsize = (8,6) ):
        """Plots the recorded metrics"""

        num_vals = np.min([len(metric) for metric in self._metrics])
        xvals = np.arange(num_vals)

        fig, axes = plt.subplots(len(self._metrics), 1, figsize = figsize, dpi = 144)

        for i, ax in enumerate(axes.reshape(-1)):

            if i == 0:
                ax.set_title('{}'.format(self._network.name))

            ax.plot(xvals[:num_vals], self._metrics[i][:num_vals], lw = 0.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(self.metric_names[i])

        plt.tight_layout()
        plt.show()


        
    def gen_plot(self, predicted_values, actual_values, itx):
        """Create a prediction plot and save to byte string. For tensorboard images tab."""

        prediction_unscaled = itx(predicted_values)
        actual_unscaled = itx(actual_values)

        sorting_idx = np.argsort(actual_unscaled.T[0])

        fig, ax = plt.subplots(figsize = (5, 3), dpi = 144)

        ax.plot(prediction_unscaled.T[0][sorting_idx],
                linestyle = 'none', marker = '.', markersize = 1,
                color = 'darkblue')

        ax.plot(actual_unscaled.T[0][sorting_idx],
                linestyle = 'none', marker = '.', markersize = 1, alpha = 0.50,
                color = '#E50000')       

        ax.set_title('std: %.9f' %np.std(prediction_unscaled.T[0][sorting_idx] - actual_unscaled.T[0][sorting_idx]))

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi = 144)
        plt.close(fig)
        buf.seek(0)

        return buf.getvalue()
    
    