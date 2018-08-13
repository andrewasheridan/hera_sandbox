"""Data_Creator

base class for data creators
"""

import sys, os
from data_manipulation import *

import numpy as np

class Data_Creator(object):
    """Data_Creator - Parent for network-style specific creators

        Generates data, in an alternate thread, for training.
        Child MUST override _gen_data()
    
        Args:
            num_flatnesses(int): number of flatnesses used to generate data.
                                -   Number of data samples = 60 * num_flatnesses
            bl_data : data source. Output of get_seps_data()
            bl_dict (dict) - Dictionary of seps with bls as keys. An output of get_or_gen_test_train_red_bls_dicts()
            gains(dict): - Gains for this data. An output of load_relevant_data()
    
    ## usage:
    ## data_maker = Data_Creator_R(num_flatnesses=250, mode = 'train')
    ## data_maker.gen_data() #before loop
    ## inputs, targets = data_maker.get_data() #start of loop
    ## data_maker.gen_data() #immediately after get_data()
    
    """

    def __init__(self,
                 num_flatnesses,
                 bl_data = None,
                 bl_dict = None,
                 gains = None,
                 abs_min_max_delay = 0.040):     
        """__init__
        
        Args:
            num_flatnesses (int): Number of flatnesses to generate
            bl_data (None, optional): Description
            bl_dict (None, optional): Description
            gains (None, optional): Description
            abs_min_max_delay (float, optional): Description
        """
        self._num = num_flatnesses
                    
        self._bl_data = bl_data
        self._bl_data_c = None
        
        self._bl_dict = bl_dict
        
        self._gains = gains
        self._gains_c = None
        
        self._epoch_batch = []
        self._nu = np.arange(1024)
        self._tau = abs_min_max_delay
        
    def _gen_data(self):
        """Summary
        """
        print('Must override _gen_data()')
 
    def gen_data(self):
        """Starts a new thread and generates data there.
        """
        
        self._thread = Thread(target = self._gen_data, args=())
        self._thread.start()

    def get_data(self, timeout = 10):
        """Retrieves the data from the thread.
        
        Returns:
            list of shape (num_flatnesses, 60, 1024)
             - needs to be reshaped for training
        
        Args:
            timeout (int, optional): Description
        """
        
        if len(self._epoch_batch) == 0:
            self._thread.join(timeout)
            
        return self._epoch_batch.pop(0)