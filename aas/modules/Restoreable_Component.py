# Restoreable_Component

from pprint import pprint
import numpy as np
import sys
import os


class Restoreable_Component(object):
    """Allows saving and loading of parameters.
       Excludes saving and loading of parameters begining with '_' or parameters that refer to tensorflow objects.
       Retains the name of the initial object (will not load the name parameter from disk).
       
       Used for networks and network-trainers to ease recordkeeping."""
    
    def __init__(self, name, log_dir = 'logs/', verbose = True):
        self.name = name
        self.log_dir = log_dir
        self._vprint = sys.stdout.write if verbose else lambda *a, **k: None
        self._msg = ''
        
        
        
    def _gen_params_dict(self):
        """Excludes saving and loading of parameters begininning with '_' or parameters that refer to tensorflow objects."""
        d = self.__dict__
        return {key : d[key] for key in d.keys() if key[0] != '_' if 'tensorflow' not in str(type(d[key]))}
    
    def print_params(self):
        """Prints parameters"""
        pprint(self._gen_params_dict())

    def load_params(self, path):
        """Load in parameters from npz file. Keeps the current name."""
    
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
        """Save the parameters."""
        
        self._msg += '\rsaving parameters';self._vprint(self._msg)
        
        direc = self.log_dir + self.name + '/params/' if direc == None else direc
        if not os.path.exists(direc):
            
            self._msg += '- creating new directory';self._vprint(self._msg)
            
            os.makedirs(direc)
        np.savez(direc + self.__class__.__name__, self._gen_params_dict())  
        
        self._msg += ' - params saved';self._vprint(self._msg)