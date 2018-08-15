"""RestoreableComponent
"""

from pprint import pprint
import numpy as np
import sys
import os


class RestoreableComponent(object):
    """RestoreableComponent
    
    Allows saving / loading of object parameters.
    Excludes saving / loading of parameters begining with '_' or parameters that refer to tensorflow objects.
    Used for networks and network-trainers to ease recordkeeping.
    
    Args:
        name (str): Name of the object
        log_dir (str, optional): location to store object parameters.
        verbose (bool, optional): Be verbose
    
    Methods:
        print_params(): Print the parameters for the object
        save_params(): Saves parameters in log_dir/name/params/object_class.npz
        load_params(path): Loads parameters from path into existing object
                            - Retains the name of the existing object       
    
    Attributes:
        log_dir (str): store params in log_dir/params/CLASS_NAME
        name (str): Name of object
    """
    
    def __init__(self, name, log_dir = 'logs/', verbose = True):
        """__init__
        
        Args:
            name (str): Name of object
            log_dir (str, optional): store params in log_dir/params/CLASS_NAME
            verbose (bool, optional): Be verbose
        """
        self.name = name
        self.log_dir = log_dir
        self._vprint = sys.stdout.write if verbose else lambda *a, **k: None # for verbose printing
        self._msg = ''
        
    def _gen_params_dict(self):
        """_gen_params_dict

        Generate dict of class attributes.
        Excludes parameters begininning with '_' or parameters that refer to tensorflow objects.
        
        Returns:
            dict: keys are attribute names, values are their values
        """
        d = self.__dict__
        return {key : d[key] for key in d.keys() if key[0] != '_' if 'tensorflow' not in str(type(d[key]))}
    
    def print_params(self):
        """Prints restoreable parameters"""
        pprint(self._gen_params_dict())

    def load_params(self, path):
        """load_params

        Load in parameters from npz file.
        Do not include the .npz suffix in path
        Keeps the current object name.
        
        Args:
            path (str): path to restoreabale parameters
        """
    
        self._msg = '\rloading parameters';self._vprint(self._msg)
        
        a = np.load(path + '.npz')
        d = dict(zip(("data1{}".format(k) for k in a), (a[k] for k in a)))
        
        name = self.name
        params = d['data1arr_0'][()]
        for key in params:
            setattr(self, key, params[key])
        self.name = name  
        
        self._msg = '\rparams loaded';self._vprint(self._msg)
        
    def save_params(self, direc = None):
        """save_params

        Saves the restoreable parameters.
        Setting direc will override the default location of log_dir/params/
        
        Args:
            direc (str, optional): storeage directory
        """
        
        self._msg += '\rsaving parameters';self._vprint(self._msg)
        
        direc = self.log_dir + self.name + '/params/' if direc is None else direc
        if not os.path.exists(direc):
            
            self._msg += '- creating new directory';self._vprint(self._msg)
            
            os.makedirs(direc)
        np.savez(direc + self.__class__.__name__, self._gen_params_dict())  
        
        self._msg += ' - params saved';self._vprint(self._msg)