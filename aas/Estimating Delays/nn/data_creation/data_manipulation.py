"""data_manipulation

Collection of data helper files
"""
import numpy as np
from pyuvdata import UVData
import hera_cal as hc
import random
from threading import Thread
from glob import glob
import os

def load_relevant_data(miriad_path, calfits_path):
    """Loads redundant baselines, gains, and data.
    
    Arguments:
        miriad_path (str): path to a miriad file for some JD
        calfits_path (str): path to a calfits file for the same JD
    
    Returns:
        red_bls, gains, uvd
    
    """

    # read the data
    uvd = UVData()
    uvd.read_miriad(miriad_path)

    # get the redundancies for that data
    aa = hc.utils.get_aa_from_uv(uvd)
    info = hc.omni.aa_to_info(aa)
    red_bls = np.array(info.get_reds())

    # gains for same data 
    gains, _ = hc.io.load_cal(calfits_path)
    
    return red_bls, gains, uvd

def get_good_red_bls(red_bls, gain_keys, min_group_len = 4):
    """get_good_red_bls

    Select all the good antennas from red_bls
    
    Each baseline group in red_bls will have its bad separations removed.
    Groups with less than min_group_len are removed.
    
    Arguments:
        red_bls (list of lists of tuples of ints)
             - Each sublist is a group of redundant separations for a unique baseline.
        gain_keys (dict): gains.keys() from hc.io.load_cal()
        min_group_len (int, optional): Minimum number of separations in a 'good' sublist.
            - Default = 4, so that both training and testing can take two seps)
    
    Returns:
        list of lists: Each sublist is a len >=4 list of separations of good antennas
    
    """
    
    def ants_good(sep):
        """ants_good

        Returns True if both antennas are good.
        
        Because we are using data from firstcal
        (is this right? I have trouble rememebring the names
        and properties of the different data sources)
        we can check for known good or bad antennas by looking to see if the antenna
        is represented in gains.keys(). If the antenna is present, its good.
        
        Arguments:
            sep (tuple of ints): antenna indices
        
        Returns:
            bool: True if both antennas are in gain_keys else False
        """

        ants = [a[0] for a in gain_keys]

        if sep[0] in ants and sep[1] in ants:
            return True

        else:
            return False

    good_redundant_baselines = []
    
    for group in red_bls:
        
        new_group = []
        
        for sep in group:
            
            # only retain seps made from good antennas
            if ants_good(sep) == True:
                new_group.append(sep)
                
        new_group_len = len(new_group)
        
        # make sure groups are large enough that both the training set
        # and the testing set can take two seps 
        if new_group_len >= min_group_len:
            
            good_redundant_baselines.append(sorted(new_group))
            
    return good_redundant_baselines

def _train_test_split_red_bls(red_bls, training_percent = 0.80):
    """_train_test_split_red_bls

    Split a list of redundant baselines into a training set and a testing set. 
    
    Each of the two sets has at least one pair of baselines from every group from red_bls.
    However, separations from one set will not appear the other.
    
    Arguments:
        red_bls (list of lists): Each sublist is a group of redundant separations
                                  for a unique baseline.
                                  Each sublist must have at least 4 separations.
        training_percent (float, optional): - *Approximate* portion of the separations that will
                                   appear in the training set.
    
    Returns:
        dict, dict :train_red_bls_dict, test_red_bls_dict
    
    """
    
    
    training_redundant_baselines_dict = {}
    testing_redundant_baselines_dict = {}

    # make sure that each set has at least 2 seps from each group
    #
    thinned_groups_dict = {}
    for group in red_bls:

        # group key is the sep with the lowest antenna indicies
        key = sorted(group)[0]
        
        # pop off seps from group and append them to train or test groups
        random.shuffle(group)
        training_group = []
        training_group.append(group.pop())
        training_group.append(group.pop())

        testing_group = []
        testing_group.append(group.pop())
        testing_group.append(group.pop())
        
        # add the new train & test groups into the dicts
        training_redundant_baselines_dict[key] = training_group
        testing_redundant_baselines_dict[key] = testing_group
        
        # if there are still more seps in the group, save them into a dict for later assignment
        if len(group) != 0:
            thinned_groups_dict[key] = group

    # Shuffle and split the group keys into two sets using 
    #
    thinned_dict_keys = thinned_groups_dict.keys()
    random.shuffle(thinned_dict_keys)
    
    # Because we are ensuring that each set has some seps from every group,
    # the ratio of train / test gets reduced a few percent.
    # This (sort of) accounts for that with an arbitrary shift found by trial and error.
    #    
    #    Without this the a setting of training_percent = 0.80 results in a 65/35 split, not 80/20.
    #    
    #    I assume there is a better way...
    t_pct = np.min([0.95, training_percent + 0.15])

    
    # these 'extra' keys are the keys that each set will extract seps from thinned_groups_dict with
    training_red_bls_extra, testing_red_bls_extra = np.split(thinned_dict_keys,
                                                             [int(len(thinned_dict_keys)*t_pct)])

    # extract seps from thinned_groups_dict and apply to same key in training set
    for key in training_red_bls_extra:
        key = tuple(key)
        group = thinned_groups_dict[key]
        training_group = training_redundant_baselines_dict[key]
        training_group.extend(group)
        training_redundant_baselines_dict[key] = training_group

    # extract seps from thinned_groups_dict and apply to same key in testing set
    for key in testing_red_bls_extra:
        key = tuple(key)
        group = thinned_groups_dict[key]
        testing_group = testing_redundant_baselines_dict[key]
        testing_group.extend(group)
        testing_redundant_baselines_dict[key] = testing_group
        
    return training_redundant_baselines_dict, testing_redundant_baselines_dict

def _loadnpz(filename):
    """_loadnpz

    Loads up npzs. For dicts do loadnpz(fn)[()]
    
    Args:
        filename (str): path to npz to load
    
    Returns:
        numpy array: the data in the npz
    """
    
    a = np.load(filename)
    d = dict(zip(("data1{}".format(k) for k in a), (a[k] for k in a)))
    
    return d['data1arr_0']

def get_or_gen_test_train_red_bls_dicts(red_bls = None,
                                        gain_keys = None,
                                        training_percent = 0.80,
                                        training_load_path = None,
                                        testing_load_path = None):
    """get_or_gen_test_train_red_bls_dicts

    Create or generate baseline dicts for training and testing
    
    Args:
        red_bls (list of lists of tuples of ints): List of redundant baselines
        gain_keys (list of int): antenna gains
        training_percent (float, optional): Approximate portion of data to assign as training data
        training_load_path (None, optional): Path to prepared dict of baselines
        testing_load_path (None, optional): Path to prepared dict of baselines
    
    Returns:
        dict, dict: training_red_bls_dict, testing_red_bls_dict
    """
    training_fn = 'data/training_redundant_baselines_dict_{}'.format(int(100*training_percent))
    testing_fn = 'data/testing_redundant_baselines_dict_{}'.format(int(100*training_percent))

    if training_load_path != None and testing_load_path != None:
        
        training_red_bls_dict = _loadnpz(training_load_path)[()]
        testing_red_bls_dict = _loadnpz(testing_load_path)[()]

    else:

        # if the files exist dont remake them.
        if os.path.exists(training_fn + '.npz') and os.path.exists(testing_fn + '.npz'):
            training_red_bls_dict = _loadnpz(training_fn + '.npz')[()]
            testing_red_bls_dict = _loadnpz(testing_fn + '.npz')[()]

        else:
        
            assert type(red_bls) is not None, "Provide a list of redundant baselines"
            assert type(gain_keys) is not None, "Provide a list of gain keys"
            
            good_red_bls = get_good_red_bls(red_bls, gain_keys)
            training_red_bls_dict, testing_red_bls_dict = _train_test_split_red_bls(good_red_bls,
                                                                                    training_percent = training_percent)

            np.savez(training_fn, training_red_bls_dict)
            np.savez(testing_fn, testing_red_bls_dict)

    return training_red_bls_dict, testing_red_bls_dict

def get_seps_data(red_bls_dict, uvd):
    """get_seps_data

    Get the data for all the seps in a redundant baselines dictionary.
    
    Args:
        red_bls_dict (dict): leys are unique baselines, values are tuples of antenna indicies
        uvd (pyuvdata.UVData: 
    
    Returns:
        dict: keys are separations (int, int), values are complex visibilitoes
    """
    
    data = {}
    for key in red_bls_dict.keys():
        for sep in red_bls_dict[key]:
            data[sep] = uvd.get_data(sep)
    
    return data

