"""estdel - for estimating delays
 - Andrew Sheridan sheridan@berkeley.edu

Estimate the overcall cable delay in visibility ratios.

For each list of 60 x 1024 complex visibility ratios produce 60 estimated cable delays.

Passes each row through one of (or both of) two trained neural networks. 

Sign network classifies the sign of the delay as being positive or negative.
Magnitude network classfies the magnitude of the delay as being in one of 401 classes.
    -   401 classes from 0.00 to 0.0400, each of width 0.0001
    -   default conversion to 401 classes from 0 to ~ 400 ns

VratioDelaySign estimates the sign of the delay (< 0 or >= 0)
VratioDelayMagnitude estimates the magnitude of the delay
VratioDelay provides Delay_Sign * Delay_Magnitude

tau = # slope in range -0.0400 to 0.0400
nu = np.arange(1024) # unitless frequency channels
phi = # phase in range 0 to 1
data = np.exp(-2j*np.pi*(nu*tau + phi))

estimator = estdel.VratioDelay(data)
prediction = estimator.predict()

# prediction should output tau
"""

import pkg_resources
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.WARN)

_TRAINED_MODELS_DIR = 'trained_models'

# fn for best postive-negative classifier
_SIGN_PATH = 'sign_NN_frozen.pb'

# fn for best magnitude classifier
_MAG_PATH = 'mag_NN_frozen.pb'





class _DelayPredict(object):
    """_DelayPredict
    
    Base class for predictions.

    Data processing and predictions.
    
    """
    
    def __init__(self, data):
        """__init__
        
        Constants:

            _n_freqs = 1024
        """
        self._data = data
        self._n_freqs = 1024


    def _angle_tx(self, x):
        """_angle_tx

        Scales (-pi, pi) to (0, 1)
        
        Args:
            x (numpy array of floats): Angle data in range -pi to pi
        
        Returns:
            array: Description
        """

        return (x + np.pi) / (2. * np.pi)
    

    def _preprocess_data(self):
        """_preprocess_data

            Converts data from complex to real (angle), scales, and reshapes
        
        Returns:
            TYPE: Description
        """
        return self._angle_tx(np.angle(self._data)).reshape(-1,1,self._n_freqs,1)


    def _predict(self):
        """_predict

        Import frozen tensorflow network, activate graph, 
        feed data, and make prediction.

        """

        resource_package = __name__ 
        resource_path = '/'.join((_TRAINED_MODELS_DIR, self._model_path))
        path = pkg_resources.resource_filename(resource_package, resource_path)

        with tf.gfile.GFile(path, "rb") as f:
            restored_graph_def = tf.GraphDef()
            restored_graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(restored_graph_def, input_map=None, return_elements=None, name="")

            sample_keep_prob = graph.get_tensor_by_name('keep_probs/sample_keep_prob:0')
            conv_keep_prob = graph.get_tensor_by_name('keep_probs/conv_keep_prob:0')
            is_training = graph.get_tensor_by_name('is_training:0')
            X = graph.get_tensor_by_name('sample/X:0')

            # add hook to output operation
            pred_cls = graph.get_tensor_by_name('predictions/ArgMax:0')

        with tf.Session(graph=graph) as sess:
            feed_dict = {sample_keep_prob : 1.,
                         conv_keep_prob : 1.,
                         is_training : False,
                         X: self.data}

            # collect prediction
            self._pred_cls = sess.run(pred_cls, feed_dict = feed_dict)

            sess.close()




class VratioDelaySign(_DelayPredict):
    """VratioDelaySign
    
    Estimates visibility ratio cable delay sign by using two pretrained neural networks.
    
    Methods:
        predict()
            - call to make sign prediction for data
    
    Attributes:
        data (numpy array of floats): Input data of redundant visibility ratios is processed for predictions
        predictions (numpy array of floats): The converted raw magnitude predictions (see predict())
        raw_predictions (list of floats): The raw magnitude predictions from the network
    """

    def __init__(self, data):
        """__init__

        Preprocesses data for prediction.
        
            - converts complex data to angle
            - scales angles to range preferred by networks
            - reshapes 2D data to 4D tensor
        
        
        Args:
            data (list of complex): shape = (N, 1024)
                - redundant visibility ratios
        
        """
        _DelayPredict.__init__(self, data = data)

        self.data = self._preprocess_data()
        self._model_path = _SIGN_PATH


    def _pred_cls_to_sign(self):
        """_pred_cls_to_sign
        
            Convert index of predicted class index to value
              1 if class is postive
             -1 if class is negative

        Returns:
            list of ints: 
        """
        return [1 if x == 0 else -1 for x in self._pred_cls]
         
    

    def predict(self):
        """predict
        
        Returns:
            numpy array of floats: sign predictions
        """
        self._predict()
        self.raw_predictions = self._pred_cls_to_sign()
        self.predictions = np.array(self.raw_predictions)

        return self.predictions


def _default_conversion_fn(x):
    """_default_conversion_fn

    Convert unitless predictions to nanoseconds
    Based on 1024 channels and 
    0.100 GHz - 0.200 GHz range
    
    Args:
        x (numpy array of floats): Predicted values in range 0.000 to 0.040
    
    Returns:
        numpy array of floats: Converted predicted value
    """
    
    freqs = np.linspace(0.100,0.200,1024) 
    channel_width_in_GHz = np.mean(np.diff(freqs))

    return np.array(x) / channel_width_in_GHz




class VratioDelayMagnitude(_DelayPredict):
    """VratioDelayMagnitude
    
    Estimates visibility ratio total cable delay by using two pretrained neural networks.
    
    Methods:
        predict()
            - call to make prediction
        
    Attributes:
        data (list of complex floats): Visibility ratios
        predictions (numpy array of floats): The converted raw magnitude predictions (see predict())
        raw_predictions (list of floats): The raw magnitude predictions from the network
    """

    def __init__(self, data):
        """Preprocesses data for prediction.
        
            - converts complex data to angle
            - scales angles to range preferred by networks
            - reshapes 2D data to 4D tensor
        
        
        Args:
            data (list of complex floats): shape = (N, 1024)
                - redundant visibility ratios
        
        """
        _DelayPredict.__init__(self, data = data)

        self.data = self._preprocess_data()
        self._model_path = _MAG_PATH


    def _pred_cls_to_magnitude(self):
        """Summary

        Convert predicted label index to magnitude
        
        Returns:
            list of floats: Magnitudes
        """
        magnitudes = np.arange(0,0.04 + 0.0001, 0.0001)
        return [magnitudes[x] for x in self._pred_cls]
    

    def predict(self, conversion_fn='default'):
        """predict
        
        Args:
            conversion_fn (None, str, or function):
                - None - Do no conversion, output predictions are the raw predictions
                - 'default' - convert raw predictions to ns by using frequencies 
                    with a 100MHz range over 1024 channels
                - OR provide your own function to do the conversion
                    - one required argument, the raw predictions
        
        Returns:
            numpy array of floats or list of floats: predictions
        """
        self._conversion_fn = conversion_fn
        self._predict()
        self.raw_predictions = self._pred_cls_to_magnitude()

        if self._conversion_fn is None:
            
            self.predictions = self.raw_predictions

        if self._conversion_fn == 'default':

            self.predictions = _default_conversion_fn(self.raw_predictions)

        else:

            self.predictions = self._conversion_fn(self.raw_predictions)

        return np.array(self.predictions)


class VratioDelay(object):
    """VratioDelay
    
    Estimates visibility ratio total cable delay by using two pretrained neural networks.
    
    Methods:
        predict()
            - call to make prediction
    
    Arrtributes:
        raw_predictions (list of floats): The raw predictions from the network
        predictions (numpy array of floats or list of floats) = The converted raw predictions

    """
    
    def __init__(self, data):
        """__init__

        Preprocesses data for prediction.
        
            - converts complex data to angle
            - scales angles to range preferred by networks
            - reshapes 2D data to 4D tensor
        
        
        Args:
            data (list of complex floats): shape = (N, 1024)
                - redundant visibility ratios
        
        """
        
        self._mag_evaluator = VratioDelayMagnitude(data)
        self._sign_evaluator = VratioDelaySign(data)
    

    def predict(self, conversion_fn='default'):
        """predict

            Make predictions
        
        Args:
            conversion_fn (str, optional): Description
            conversion_fn (None, 'default', or function
                - None - Do no conversion, output predictions are the raw predictions
                - 'default' - convert raw predictions to ns by using frequencies 
                    with a 100MHz range over 1024 channels
                - OR provide your own function to do the conversion
                    - takes in one argument, the raw predictions
        
        
        
        Returns:
            numpy array of floats or list of floats: Predicted values
        """
        signs = self._sign_evaluator.predict()
        mags = self._mag_evaluator.predict(conversion_fn=conversion_fn)
        self.raw_predictions = [self._mag_evaluator.raw_predictions[i]*signs[i] for i in range(len(signs))]
        self.predictions = signs*mags
        return self.predictions




class DelaySolver(object):
    """DelaySolver
    
    Args:
        list_o_sep_pairs (list of lists of tuples): shape = (N, 2, 2)
            ex: list_o_sep_pairs[0] = [(1, 2), (3, 4)]
            each length 2 sublist is made of two redundant separations, one in each tuple
    
        v_ratios (list of complex floats): shape = (N, 60, 1024)

            Complex visibility ratios made from the the corresponding redundant sep pairs in list_o_sep_pairs
    
        true_ant_delays (dict ): dict of delays with antennas as keys,
            ex : true_ant_delays[143] = 1.2
            if conversion_fn == 'default', ant delays should be in ns
    
    Attributes:
        A (numpy array of ints): The matrix representing the redundant visibility ratios
        b (numpy array of floats): A times x
        unique_ants (numpy array of ints): All the unique antennas in list_o_sep_pairs
        v_ratio_row_predictions (numpy array of floats or list of floats): Predicted values
        v_ratio_row_predictions_raw (list of floats): Predicted values with no conversion
        x (list floats): True delays in order of antenna
    """
    
    def __init__(self,
                 list_o_sep_pairs,
                 v_ratios,
                 conversion_fn='default',
                 true_ant_delays={}, # dict {ant_idx : delay}
                ):
        """__init__

        Preprocess data, make predictions, covert data to ns, 
        construct A matrix.

        """
        self._list_o_sep_pairs = list_o_sep_pairs # shape = (N, 2, 2)
        self._v_ratios = v_ratios # complex, shape = (N, 60, 1024) # will be reshaped to (-1, 1, 1024, 1)
        self._true_ant_delays = true_ant_delays
        self._conversion_fn = conversion_fn

        self.unique_ants = np.unique(list_o_sep_pairs)
        self._max_ant_idx = np.max(self.unique_ants) + 1 
                                    
        self._make_A_from_list_o_sep_pairs()

        self._predictor = VratioDelay(v_ratios)
        self._predictor.predict(conversion_fn=conversion_fn)

        self.v_ratio_row_predictions = self._predictor.predictions
        self.v_ratio_row_predictions_raw = self._predictor.raw_predictions
        


        

        
        

    def _get_A_row(self, sep_pair):
        
        # sep_pair like [(1,2), (2,3)]
        a = np.array(list(sum(sep_pair, ())))

        # construct the row
        # https://stackoverflow.com/a/29831596

        # row is 4 x _max_ant_idx, all zeros
        row = np.zeros((a.size, self._max_ant_idx), dtype = int)

        # for each element in sep_pair, got to the corresponding row
        # and assign the corresponding antenna the value 1
        row[np.arange(a.size),a] = 1 

        # flip the sign of the middle two rows
        row[1] *= -1
        row[2] *= -1

        # add the rows, row is now 1 x _max_ant_idx
        row = np.sum(row, axis = 0)

        return row


    def _make_A_from_list_o_sep_pairs(self):

        self.A = []
        for sep_pair in self._list_o_sep_pairs:

            # each visibility ratio of height 60 has one sep_pair
            # so make 60 rows for each
            # so that A is the correct shape
            # (because the prediction will output a unique prediction 
            # for each row in the visibility ratio)
            self.A.append(np.tile(self._get_A_row(sep_pair), (60,1)))

        self.A =  np.asarray(self.A).reshape(-1, self._max_ant_idx)


    def true_b(self):

        
        self.x = [self._true_ant_delays[ant] for ant in self.unique_ants]
        self.b = np.matmul(self.A[:, self.unique_ants], self.x)

        return self.b


