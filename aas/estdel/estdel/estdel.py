"""                                                
                         _/            _/            _/   
    _/_/      _/_/_/  _/_/_/_/    _/_/_/    _/_/    _/    
 _/_/_/_/  _/_/        _/      _/    _/  _/_/_/_/  _/     
_/            _/_/    _/      _/    _/  _/        _/      
 _/_/_/  _/_/_/        _/_/    _/_/_/    _/_/_/  _/       
                                                          
 
estdel - for estimating delays

uses two trained neural networks to compute the 
slope of the phase angle of wrapped complex data

Delay_Sign estimates the sign of the delay (< 0 or >= 0)
Delay_Magnitude estimates the magnitude of the delay
Cable_Delay provides Delay_Sign * Delay_Magnitude

tau = # slope in range -0.0400 to 0.0400
nu = np.arange(1024) # unitless frequency channels
phi = # phase in range 0 to 1
data = np.exp(-2j*np.pi*(nu*tau + phi))

estimator = estdel.Cable_Delay(data)
prediction = estimator.predict()

# prediction should output tau
"""


import numpy as np
import tensorflow as tf
from CNN_DS_BN_C import CNN_DS_BN_C

tf.logging.set_verbosity(tf.logging.WARN)

# path to best postive-negative classifier
PN_PATH = '../../logs/CNN_DS_BN_C_2_401_aug7_POSNEG/trained_model.ckpt-5490'

# path to best magnitude classifier
MAG_PATH = '../../logs/CNN_DS_BN_C_2_401_aug7_e/trained_model.ckpt-1000'

class DelayPredict(object):
    """ DelayPredict

        Base class for predictions using CNN_DS_BN_C networks

    """
    
    def __init__(self, data):
        
        self._data = data
        self._n_freqs = 1024

    def _angle_tx(self, x):
        # assumes x is numpy array
        # scales (-pi,pi) to (0,1)
        return (x + np.pi) / (2. * np.pi)
    
    def _preprocess_data(self):
        return self._angle_tx(np.angle(self._data)).reshape(-1,1,self._n_freqs,1)

    
    def _predict(self):  
        # construct a network graph
        # load in pretrained variables
        # return predicted class index
        self._network.create_graph()
        
        saver = tf.train.Saver()
        with tf.Session() as session:

            saver.restore(session, self._model_path)
            
            feed_dict = {self._network.X: self.data,
                         self._network.sample_keep_prob : 1.,
                         self._network.conv_keep_prob : 1.,
                         self._network.is_training : False}

            self._pred_cls = session.run([self._network.pred_cls], feed_dict = feed_dict)
            session.close()

class Delay_Sign(DelayPredict):

    def __init__(self, data, verbose=False):
        
        DelayPredict.__init__(self, data = data)
        
        self._network = CNN_DS_BN_C('sign_eval', 3, 2, verbose=verbose)
        self.data = self._preprocess_data()
        self._model_path = PN_PATH

    def _pred_cls_to_sign(self):
        # convert predicted class index to value
        # self._pred_cls has 0 for positive and 1 for negative
        self.pred_signs = [1 if x == 0 else -1 for x in self._pred_cls[0]]
         
        return self.pred_signs
    
    def predict(self):
        """ predict

            Returns:
                list of sign predictions
        """
        self._predict()
        return self._pred_cls_to_sign()

class Delay_Magnitude(DelayPredict):

    def __init__(self, data, verbose=False):
        
        DelayPredict.__init__(self, data = data)
        
        self._network = CNN_DS_BN_C('mag_eval', 3, 401, verbose=verbose)
        self.data = self._preprocess_data()
        self._model_path = MAG_PATH

    def _pred_cls_to_magnitude(self):
        # convert predicted class index to value
        magnitudes = np.arange(0,0.04 + 0.0001, 0.0001)
        self.pred_mags = [magnitudes[x] for x in self._pred_cls[0]]
         
        return self.pred_mags
    
    def predict(self):
        """ predict

            Returns:
                list of magnitude predictions
        """
        self._predict()
        return self._pred_cls_to_magnitude()

class Cable_Delay(object):
    """ Cable_Delay

    Estimates cable delay by using two pretrained neural networks.

    Methods:
        predict()
            - call to make prediction
    """
    
    def __init__(self, data, verbose=False):
        """ Preprocesses data for prediction.

            - converts complex data to angle
            - scales angles to range preferred by networks
            - reshapes 2D data to 4D tensor
    

        Args:

            data : list, complex, shape = (N, 1024)
                - redundant visibility ratios
            verbose : bool - be verbose
        """
        
        self._mag_evaluator = Delay_Magnitude(data, verbose=verbose)
        self._sign_evaluator = Delay_Sign(data, verbose=verbose)
    
    def predict(self):
        """ predict

            Returns:
                list of predictions
        """
        self.signs = self._sign_evaluator.predict()
        self.mags = self._mag_evaluator.predict()
        
        return np.array(self.signs)*np.array(self.mags)