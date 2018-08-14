"""Data_Creator

base class for data creators
"""

import sys, os
from data_manipulation import *

import numpy as np

class Data_Creator(object):
    """Data_Creator - Parent for network style-specific creators

        Generates data, in an alternate thread, for training.
        Child MUST implement _gen_data()
    
        Args:
            num_flatnesses(int): number of flatnesses used to generate data.
                                -   Number of data samples = 60 * num_flatnesses
            bl_data : data source. Output of get_seps_data()
            bl_dict (dict) - Dictionary of seps with bls as keys. An output of get_or_gen_test_train_red_bls_dicts()
            gains(dict): - Gains for this data. An output of load_relevant_data()
    
    """

    def __init__(self,
                 num_flatnesses,
                 bl_data,
                 bl_dict,
                 gains,
                 abs_min_max_delay = 0.040):     
        """__init__
        
        Args:
            num_flatnesses (int): Number of flatnesses to generate
            bl_data (dict): keys are baselines, data is complex visibilities
            bl_dict (dict): keys are unique redundant baseline, values are baselines in that redudany group
            gains (dict): complex antenna gains
            abs_min_max_delay (float, optional): MinMax value of delay
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
        """_gen_data

        This method is called in a separate thread via gen_data().
        Must be overridden
        """
        print('Must implement _gen_data()')
 
    def gen_data(self):
        """gen_data

        Starts a new thread and generates data there.
        """
        
        self._thread = Thread(target = self._gen_data, args=())
        self._thread.start()

    def get_data(self, timeout = 10):
        """get_data

        Retrieves the data from the thread.
  
        Args:
            timeout (int, optional): data generation timeout in seconds
        
        Returns:
            (list of complex floats): shape = (num_flatnesses, 60, 1024)
             - needs to be reshaped for training

        """
        
        if len(self._epoch_batch) == 0:
            self._thread.join(timeout)
            
        return self._epoch_batch.pop(0)