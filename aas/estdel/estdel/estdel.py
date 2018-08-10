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
nu = _np.arange(1024) # unitless frequency channels
phi = # phase in range 0 to 1
data = _np.exp(-2j*_np.pi*(nu*tau + phi))

estimator = estdel.Cable_Delay(data)
prediction = estimator.predict()

# prediction should output tau
"""


import numpy as _np
import tensorflow as _tf

_tf.logging.set_verbosity(_tf.logging.WARN)

# path to best postive-negative classifier
_SIGN_PATH = 'trained_models/sign_NN_frozen.pb'

# path to best magnitude classifier
_MAG_PATH = 'trained_models/mag_NN_frozen.pb'

class _DelayPredict(object):
    """ _DelayPredict

        Base class for predictions

    """
    
    def __init__(self, data):
        
        self._data = data
        self._n_freqs = 1024

    def _angle_tx(self, x):
        # assumes x is numpy array
        # scales (-pi,pi) to (0,1)
        return (x + _np.pi) / (2. * _np.pi)
    
    def _preprocess_data(self):
        return self._angle_tx(_np.angle(self._data)).reshape(-1,1,self._n_freqs,1)

    def _predict(self):

        with _tf.gfile.GFile(self._model_path, "rb") as f:
            restored_graph_def = _tf.GraphDef()
            restored_graph_def.ParseFromString(f.read())

        with _tf.Graph().as_default() as graph:
            _tf.import_graph_def(restored_graph_def, input_map=None, return_elements=None, name="")

            sample_keep_prob = graph.get_tensor_by_name('keep_probs/sample_keep_prob:0')
            conv_keep_prob = graph.get_tensor_by_name('keep_probs/conv_keep_prob:0')
            is_training = graph.get_tensor_by_name('is_training:0')
            X = graph.get_tensor_by_name('sample/X:0')

            # add hook to output operation
            pred_cls = graph.get_tensor_by_name('predictions/ArgMax:0')

        with _tf.Session(graph=graph) as sess:
            feed_dict = {sample_keep_prob : 1.,
                         conv_keep_prob : 1.,
                         is_training : False,
                         X: self.data}

            # collect prediction
            self._pred_cls = sess.run(pred_cls, feed_dict = feed_dict)

            sess.close()


class Cable_Delay_Sign(_DelayPredict):
    """ Cable_Delay

    Estimates cable delay by using two pretrained neural networks.

    Methods:
        predict()
            - call to make prediction

    Arrtributes:
        raw_predictions : (list of floats) - The raw sign predictions from the network
        predictions : numpy array of floats = The raw sign predictions as a numpy array 
    """
    def __init__(self, data):
        """ Preprocesses data for prediction.

            - converts complex data to angle
            - scales angles to range preferred by networks
            - reshapes 2D data to 4D tensor
    

        Args:

            data : list, complex, shape = (N, 1024)
                - redundant visibility ratios

        """
        _DelayPredict.__init__(self, data = data)

        self.data = self._preprocess_data()
        self._model_path = _SIGN_PATH

    def _pred_cls_to_sign(self):
        # convert predicted class index to value
        # self._pred_cls has 0 for positive and 1 for negative
        self.pred_signs = [1 if x == 0 else -1 for x in self._pred_cls]
         
        return self.pred_signs
    
    def predict(self):
        """ predict

            Returns:
                numpy array of sign predictions
        """
        self._predict()
        self.raw_predictions = self._pred_cls_to_sign()
        self.predictions = _np.array(self.raw_predictions)

        return self.predictions

class Cable_Delay_Magnitude(_DelayPredict):
    """ Cable_Delay

    Estimates cable delay by using two pretrained neural networks.

    Methods:
        predict()
            - call to make prediction

    Arrtributes:
        raw_predictions : (list of floats) - The raw magnitude predictions from the network
        predictions : numpy array of floats = The converted raw magnitude predictions (see predict())
    """
    def __init__(self, data):
        """ Preprocesses data for prediction.

            - converts complex data to angle
            - scales angles to range preferred by networks
            - reshapes 2D data to 4D tensor
    

        Args:

            data : list, complex, shape = (N, 1024)
                - redundant visibility ratios

        """
        _DelayPredict.__init__(self, data = data)

        self.data = self._preprocess_data()
        self._model_path = _MAG_PATH

    def _pred_cls_to_magnitude(self):
        # convert predicted class index to value
        magnitudes = _np.arange(0,0.04 + 0.0001, 0.0001)
        self.pred_mags = [magnitudes[x] for x in self._pred_cls]
         
        return self.pred_mags
    
    def predict(self, conversion_fn='default'):
        """ predict

            Args:
                conversion_fn (None, 'default', or function
                    - None - Do no conversion, output predictions are the raw predictions
                    - 'default' - convert raw predictions to ns by using frequencies 
                        with a 100MHz range over 1024 channels
                    - OR provide your own function to do the conversion
                        - one required argument, the raw predictions

            Returns:
                numpy array of predictions
        """
        self._conversion_fn = conversion_fn
        self._predict()
        self.raw_predictions = self._pred_cls_to_magnitude()

        if self._conversion_fn is None:
            
            self.predictions = self.raw_predictions

        elif self._conversion_fn == 'default':

            # this is a sorta dumb way to do this

            # 0.100 GHz - 0.200 GHz range
            freqs = _np.linspace(0.100,0.200,1024) 
            channel_width_in_GHz = _np.mean(_np.diff(freqs))

            # predictions now in nanoseconds
            self.predictions = _np.round(self.raw_predictions / channel_width_in_GHz,1)

        else:
            self.predictions = self._conversion_fn(self.raw_predictions)

        return _np.array(self.predictions)


class Cable_Delay(object):
    """ Cable_Delay

    Estimates cable delay by using two pretrained neural networks.

    Methods:
        predict()
            - call to make prediction

    Arrtributes:
        raw_predictions : (list of floats) - The raw predictions from the network
        predictions : numpy array of floats = The converted raw predictions (see predict())
    """
    
    def __init__(self, data):
        """ Preprocesses data for prediction.

            - converts complex data to angle
            - scales angles to range preferred by networks
            - reshapes 2D data to 4D tensor
    

        Args:

            data : list, complex, shape = (N, 1024)
                - redundant visibility ratios

        """
        
        self._mag_evaluator = Cable_Delay_Magnitude(data)
        self._sign_evaluator = Cable_Delay_Sign(data)
    
    def predict(self, conversion_fn='default'):
        """ predict

            Args:
                conversion_fn (None, 'default', or function(x))
                    - None - Do no conversion, output predictions are the raw predictions
                    - 'default' - convert raw predictions to ns by using frequencies 
                        with a 100MHz range over 1024 channels
                    - OR provide your own function to do the conversion
                        - takes in one argument, the raw predictions



            Returns:
                numpy array of predictions
        """
        signs = self._sign_evaluator.predict()
        mags = self._mag_evaluator.predict(conversion_fn=conversion_fn)
        self.raw_predictions = [self._mag_evaluator.raw_predictions[i]*signs[i] for i in range(len(signs))]
        self.predictions = signs*mags
        return self.predictions