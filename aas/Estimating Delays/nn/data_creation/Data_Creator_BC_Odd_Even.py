# Data_Creator_BC_Odd_Even

from data_manipulation import *
from Data_Creator import Data_Creator

import numpy as np

class Data_Creator_BC_Odd_Even(Data_Creator):

    def __init__(self,
                 num_flatnesses,
                 bl_data = None,
                 bl_dict = None,
                 gains = None,
                 abs_min_max_delay = 0.040):

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
        
        # convert 0.0149 to 149 etc, for finding odd or even
        rounded_targets = 10000*np.asarray([np.round(abs(np.round(d * 100,2)/100), 5) for d in targets[permutation_index]]).reshape(-1)
        labels = [[1,0] if x%2 == 0 else [0,1] for x in rounded_targets]
        

        self._epoch_batch.append((angle_tx(inputs[permutation_index]), labels))

