"""Summary
"""
# Data_Creator_C

from data_manipulation import *
from Data_Creator import Data_Creator

import numpy as np

class Data_Creator_C(Data_Creator):
    """Data_Creator_C C = Classification
    
    Creates one dimensional inputs and their labels for training
    Inputs are simulated angle data generated with true data as their base
        - see ????.ipynb for discussion of input data
    Labels are one-hot vectors (optionally blurred vectors)
        - see ????.ipynb for discussion of labels and blurring
    
    Each input is generated:
        - a pair of redundant baselines is randomly chosen
        - the ratio of visibilties is constructed (a complex flatness, see discussion)
        - a separate target cable delay is applied to each row of the flatness
            - targets in the range (-abs_min_max_delay, abs_min_max_delay)
        - the angle is computed of the flatness with the applied delay
    
    Each label is generated:
        - the smooth range of targets is converted to a series of steps of the desired precision
            - see Classifier_Class_Explanation.ipynb for discussion
        - each unique step value is assigned a one-hot vector (from min to max)
        - each target is assigned its label target
        - if blur != 0 blur is applied
            - Imagine each one-hot vector as a delta function with area = 1
            - blur converts the delta function into a gaussian with area = 1
            - see ????.ipynb for discussion
    
    Args:
        num_flatnesses (int): number of flatnesses used to generate data.
            Number of data samples = 60 * num_flatnesses
        bl_data (dict): data source. Output of get_seps_data()
        bl_dict (dict): Dictionary of seps with bls as keys. An output of get_or_gen_test_train_red_bls_dicts()
        gains (dict): Gains for this data. An output of load_relevant_data()
        abs_min_max_delay (float, optional): targets are generated in range(-abs_min_max_delay, abs_min_max_delay)
        precision (float, optional): size of the class steps must be from (0.005,0.001,0.0005,0.00025,0.0001)
        evaluation (bool, optional): experimental
        single_dataset (bool, optional): experimental
        singleset_path (None, optional): experimental
        blur (float, optional): blur value. Convert a one-hot vector delta function to a gaussian.
            Spreads the 1.0 prob of a class over a range of nearby classes.
            blur = 0.5 spreads over about 7 classes with a peak prob of 0.5 (varies)
            blue = 0.1 spreads over about 40 classes with peak prob ~ 0.06 (varies)
            blur = 0.0, no blur
    
    Attributes:
        blur (float): Description
        evaluation (bool): experimental
        precision (float): Description
        single_dataset (bool): experimental
        singleset_path (bool): experimental
    """
    __doc__ += Data_Creator.__doc__

    def __init__(self,
                 num_flatnesses,
                 bl_data,
                 bl_dict ,
                 gains,
                 abs_min_max_delay = 0.040,
                 precision = 0.00025,
                 evaluation = False,
                 single_dataset = False,
                 singleset_path = None,
                 blur = 0.): # try 1, 0.5, 0.10
        """Summary
        
        Args:
            num_flatnesses (int): number of flatnesses used to generate data.
                Number of data samples = 60 * num_flatnesses
            bl_data (dict): data source. Output of get_seps_data()
            bl_dict (dict): Dictionary of seps with bls as keys. An output of get_or_gen_test_train_red_bls_dicts()
            gains (dict): Gains for this data. An output of load_relevant_data()
            abs_min_max_delay (float, optional): targets are generated in range(-abs_min_max_delay, abs_min_max_delay)
            precision (float, optional): size of the class steps must be from (0.005,0.001,0.0005,0.00025,0.0001)
            evaluation (bool, optional): experimental
            single_dataset (bool, optional): experimental
            singleset_path (None, optional): experimental
            blur (float, optional): blur value. Convert a one-hot vector delta function to a gaussian.
                Spreads the 1.0 prob of a class over a range of nearby classes.
                blur = 0.5 spreads over about 7 classes with a peak prob of 0.5 (varies)
                blue = 0.1 spreads over about 40 classes with peak prob ~ 0.06 (varies)
                blur = 0.0, no blur
        """
        Data_Creator.__init__(self,
                              num_flatnesses = num_flatnesses,
                              bl_data = bl_data,
                              bl_dict = bl_dict,
                              gains = gains,
                              abs_min_max_delay = abs_min_max_delay)

        self.precision = precision
        self.evaluation = evaluation
        self.single_dataset = single_dataset
        self.singleset_path = singleset_path
        self.blur = blur
        
    def _load_singleset(self):
        """_load_singleset

        load a set of specific baseline-pairs
        """
        def loadnpz(filename):
            """loadnpz
            
            Args:
                filename (str): path
            
            Returns:
                numpy array: loaded npz
            """
            a = np.load(filename)
            d = dict(zip(("data1{}".format(k) for k in a), (a[k] for k in a)))
            return d['data1arr_0']

        return [[(s[0][0],s[0][1]), (s[1][0],s[1][1])] for s in loadnpz(self.singleset_path)]
    
    def _blur(self, labels):
        """_blur

        Convert a one-hot vector delta function to a gaussian.
        Spreads the 1.0 prob of a class over a range of nearby classes.
        blur = 0.5 spreads over about 7 classes with a peak prob of 0.5
        blur = 0.1 spreads over about 40 classes with peak prob ~ 0.06
        
        Args:
            labels (list of lists of ints): class labels
        
        Returns:
            list of lists of floats: class labels, blurred
        """
        for i, label in enumerate(labels):
            true_val = np.argmax(label)
            mean = true_val; std = self.blur; variance = std**2
            x = np.arange(len(label))
            f = np.exp(-np.square(x-mean)/2*variance)
            labels[i] = f/np.sum(f)
        return labels
             
    def _gen_data(self):            
        """_gen_data

        Generates artiicial flatnesses by combining two redundant visbilites and their gains.
        Applies different known delay to each row of the floatnesses.
        Scales data for network.

        Converts numerical delay valyue to classification label
        """
        # scaling tools
        # the NN likes data in the range (0,1)
        angle_tx  = lambda x: (np.asarray(x) + np.pi) / (2. * np.pi)
        angle_itx = lambda x: np.asarray(x) * 2. * np.pi - np.pi

        delay_tx  = lambda x: (np.array(x) + self._tau) / (2. * self._tau)
        delay_itx = lambda x: np.array(x) * 2. * self._tau - self._tau
        
        targets = np.random.uniform(low = -self._tau, high = self._tau, size = (self._num * 60, 1))
        applied_delay = np.exp(-2j * np.pi * (targets * self._nu + np.random.uniform()))



        assert type(self._bl_data) != None, "Provide visibility data"
        assert type(self._bl_dict) != None, "Provide dict of baselines"
        assert type(self._gains)   != None, "Provide antenna gains"

        if self._bl_data_c == None:
            self._bl_data_c = {key : self._bl_data[key].conjugate() for key in self._bl_data.keys()}

        if self._gains_c == None:
            self._gains_c = {key : self._gains[key].conjugate() for key in self._gains.keys()}


        def _flatness(seps):
            """_flatness

            Create a flatness from a given pair of seperations, their data & their gains.
            
            Args:
                seps (list of tuples of ints): two redundant separations 
            
            Returns:
                numpy array of complex floats: visibility ratio flatnesss
            """

            a, b = seps[0][0], seps[0][1]
            c, d = seps[1][0], seps[1][1]


            return self._bl_data[seps[0]]   * self._gains_c[(a,'x')] * self._gains[(b,'x')] * \
                   self._bl_data_c[seps[1]] * self._gains[(c,'x')]   * self._gains_c[(d,'x')]

        inputs = []
        seps = []

        singleset_seps = None if self.singleset_path == None else self._load_singleset()

        for i in range(self._num):

            unique_baseline = random.sample(self._bl_dict.keys(), 1)[0]
            if self.singleset_path == None:
                two_seps = [random.sample(self._bl_dict[unique_baseline], 2)][0]
            elif self.singleset_path != None:
                two_seps = singleset_seps[i]
            inputs.append(_flatness(two_seps))
            for t in range(60):
                seps.append(two_seps)

        inputs = np.angle(np.array(inputs).reshape(-1,1024) * applied_delay)
        
        permutation_index = np.random.permutation(np.arange(self._num * 60))
        
        
#         TODO: Quadruple check that rounded_targets is doing what you think it is doing

        # rounded_targets is supposed to take the true targets and round their value
        # to the value closest to the desired precision. and the absolute.
        # pretty sure i have it working for the values below
        # classes is supposed to have all the possible unique rounded values
        
        #0.0001 precision - 401 classes
        if self.precision == 0.0001:
            rounded_targets = np.asarray([np.round(abs(np.round(d * 100,2)/100), 5) for d in targets[permutation_index]]).reshape(-1)
            classes = np.arange(0,0.04 + self.precision, self.precision)
        
        #0.00025 precision - 161 classes
        if self.precision == 0.00025:
            rounded_targets = np.asarray([np.round(abs(np.round(d * 40,2)/40), 5) for d in targets[permutation_index]]).reshape(-1)
            classes = np.arange(0,0.04 + self.precision, self.precision)

        # 0.0005 precision 81 classes
        if self.precision == 0.0005:
            rounded_targets = np.asarray([np.round(abs(np.round(d * 20,2)/20), 5) for d in targets[permutation_index]]).reshape(-1)
            classes = np.arange(0,0.04 + self.precision, self.precision)

        # 0.001 precision - 41 classes
        if self.precision == 0.001:
            rounded_targets = np.asarray([np.round(abs(np.round(d * 10,2)/10), 5) for d in targets[permutation_index]]).reshape(-1)
            classes = np.arange(0,0.04 + self.precision, self.precision)

        # for 0.005 precision there should be 9 different classes
        if self.precision == 0.005:
            rounded_targets = np.asarray([np.round(abs(np.round(d * 2,2)/2), 5) for d in targets[permutation_index]]).reshape(-1)
            classes = np.arange(0,0.04 + self.precision, self.precision)

        eye = np.eye(len(classes), dtype = int)
        classes_labels = {}
        for i, key in enumerate(classes):
            classes_labels[np.round(key,5)] = eye[i].tolist()
            
            
        labels = [classes_labels[x] for x in rounded_targets]
        labels = labels if self.blur == 0 else self._blur(labels)
        
        if self.evaluation == True:
            self._epoch_batch.append((angle_tx(inputs[permutation_index]), labels, targets[permutation_index], np.asarray(seps)[permutation_index]))
        if self.single_dataset == True:
            self._epoch_batch.append((angle_tx(inputs[permutation_index]), labels, np.asarray(seps)[permutation_index]))
        else:
            self._epoch_batch.append((angle_tx(inputs[permutation_index]), labels))
        

