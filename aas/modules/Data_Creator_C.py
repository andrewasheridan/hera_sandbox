# Data_Creator_C

from data_manipulation import *
from Data_Creator import Data_Creator

import numpy as np

class Data_Creator_C(Data_Creator):
    """Currently this groups data targets into the closest of 161 values. Each block has a label of its value.
        Im not explaining this correctly...

        The point of this was to try to use a classifier to solve the problem instead of regression.
         So I took the continuous targets and converted them into a series of discontinuous steps:

        # this should round the targets to the closest 0.00025 (2.5 ns)
         rounded_targets = np.asarray([np.round(abs(np.round(d * 40,2)/40), 5) for d in targets[permutation_index]]).reshape(-1)
          
        # should be a list of all the possible rounded targets (the different class names)
          classes = np.arange(0,0.04025, 0.00025)

        # should make a dict of one hot vector for each class
        eye = np.eye(len(classes), dtype = int)
        classes_labels = {}
        for i, key in enumerate(classes):
            classes_labels[np.round(key,5)] = eye[i].tolist()

        # should assign the appropriate one-hot vector for each target
        labels = [classes_labels[x] for x in rounded_targets]

        # and then the output of _gen_data is
        self._epoch_batch.append((angle_tx(inputs[permutation_index]), labels))



    """

    def __init__(self,
                 num_flatnesses,
                 bl_data = None,
                 bl_dict = None,
                 gains = None,
                 abs_min_max_delay = 0.040):
        
        """
        Arguments
            num_flatnesses : int - number of flatnesses used to generate data.
                                   Number of data samples = 60 * num_flatnesses
            bl_data : data source. Output of get_seps_data()
            bl_dict : dict - Dictionary of seps with bls as keys. An output of get_or_gen_test_train_red_bls_dicts()
            gains : dict - Gains for this data. An output of load_relevant_data()
            
                                   
        """
        Data_Creator.__init__(self,
                              num_flatnesses = num_flatnesses,
                              bl_data = bl_data,
                              bl_dict = bl_dict,
                              gains = gains,
                              abs_min_max_delay = abs_min_max_delay)
        
        
             
    def _gen_data(self):
        
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
            """Create a flatness from a given pair of seperations, their data & their gains."""

            a, b = seps[0][0], seps[0][1]
            c, d = seps[1][0], seps[1][1]


            return self._bl_data[seps[0]]   * self._gains_c[(a,'x')] * self._gains[(b,'x')] * \
                   self._bl_data_c[seps[1]] * self._gains[(c,'x')]   * self._gains_c[(d,'x')]

        inputs = []
        for _ in range(self._num):

            unique_baseline = random.sample(self._bl_dict.keys(), 1)[0]
            two_seps = [random.sample(self._bl_dict[unique_baseline], 2)][0]

            inputs.append(_flatness(two_seps))

        inputs = np.angle(np.array(inputs).reshape(-1,1024) * applied_delay)
        
        permutation_index = np.random.permutation(np.arange(self._num * 60))
        
        
#         TODO: Quadruple check that rounded_targets is doing what you think it is doing

        # rounded_targets is supposed to take the true targets and round there value
        # to the value closest to the desired precision.
        # pretty sure i have it working for the values below
        # classes is supposed to have all the possible unique rounded values
        
        #0.00025 precision
        # rounded_targets = np.asarray([np.round(abs(np.round(d * 40,2)/40), 5) for d in targets[permutation_index]]).reshape(-1)
        # classes = np.arange(0,0.04025, 0.00025)
        # 0.0005 precision
        # x = [np.round(abs(np.round(d * 20,2)/20), 5) for d in dels]

        # 0.001 precision
        # x = [np.round(abs(np.round(d * 10,2)/10), 5) for d in dels]
        
        # 0.005 precision - 9 blocks
        rounded_targets = np.asarray([np.round(abs(np.round(d * 2,2)/2), 5) for d in targets[permutation_index]]).reshape(-1)
        classes = np.arange(0,0.045, 0.005)
    
        # for 0.00025 precision there should be 161 different classes

        eye = np.eye(len(classes), dtype = int)
        classes_labels = {}
        for i, key in enumerate(classes):
            classes_labels[np.round(key,5)] = eye[i].tolist()
            
#         print(classes_labels)
            
        labels = [classes_labels[x] for x in rounded_targets]

        self._epoch_batch.append((angle_tx(inputs[permutation_index]), labels))


