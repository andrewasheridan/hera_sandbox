import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(1, os.path.join(sys.path[0], '../networks/'))
from CNN_DS_BN_C import CNN_DS_BN_C

tf.logging.set_verbosity(tf.logging.WARN)

# path to best postive-negative classifier
PN_PATH = '../logs/CNN_DS_BN_C_2_401_aug7_POSNEG/trained_model.ckpt-5490'

# path to best magnitude classifier
MAG_PATH = '../logs/CNN_DS_BN_C_2_401_aug7_e/trained_model.ckpt-1000'

class DelayPredict(object):
    
    def __init__(self, data):
        
        self._data = data
        self._n_freqs = 1024

    def _angle_tx(self, x):
        # scales (-pi,pi) to (0,1)
        return (x + np.pi) / (2. * np.pi)
    
    def _preprocess_data(self):
        return self._angle_tx(np.angle(self._data)).reshape(-1,1,self._n_freqs,1)

    
    def _predict(self):
        
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
        
        # self._pred_cls has 0 for positive and 1 for negative
        self.pred_signs = [1 if x == 0 else -1 for x in self._pred_cls[0]]
         
        return self.pred_signs
    
    def predict(self):
        self._predict()
        return self._pred_cls_to_sign()

class Delay_Magnitude(DelayPredict):

    def __init__(self, data, verbose=False):
        
        DelayPredict.__init__(self, data = data)
        
        self._network = CNN_DS_BN_C('mag_eval', 3, 401, verbose=verbose)
        self.data = self._preprocess_data()
        self._model_path = MAG_PATH

    def _pred_cls_to_magnitude(self):
        
        magnitudes = np.arange(0,0.04 + 0.0001, 0.0001)
        
        # self._pred_cls has index of magnitude
        self.pred_mags = [magnitudes[x] for x in self._pred_cls[0]]
         
        return self.pred_mags
    
    def predict(self):
        self._predict()
        return self._pred_cls_to_magnitude()

class Cable_Delay(object):
    
    def __init__(self, data, verbose=False):
        
        self._mag_evaluator = Delay_Magnitude(data, verbose=verbose)
        self._sign_evaluator = Delay_Sign(data, verbose=verbose)
    
    def predict(self):
        self.signs = self._sign_evaluator.predict()
        self.mags = self._mag_evaluator.predict()
        
        return np.array(self.signs)*np.array(self.mags)