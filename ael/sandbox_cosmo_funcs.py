from math import pi
from astropy.cosmology import Planck15 as cosmo
import os
import sys
import numpy as np


def jy2Tstr(f, mK=False):#, bm):
    '''Return [mK sr] / [Jy] vs. frequency (in Hz)
        f = frequencies (Hz!)
    '''
    c_cmps = 2.99792458e10   # cm/s
    k_boltz = 1.380658e-16   # erg/K
    lam = c_cmps / f   #cm
    bm = 1.0 # steradian
    fac = 1.0
    if mK:
        fac = 1e3
    return 1e-23 * lam**2 / (2 * k_boltz * bm) * fac

def dL_df(z, omega_m=0.266):
    '''[Mpc]/Hz, from Furlanetto et al. (2006)'''
    c_kms = 299792.458  #km/s
    nu21 = 1420e6
    return c_kms/(cosmo.H0.value * cosmo.efunc(z)) * (z+1)**2 / nu21

def dL_dth(z):
    '''[Mpc]/radian, from Furlanetto et al. (2006)'''
    return cosmo.comoving_distance(z).value

def dk_deta(z):
    '''2pi * [Mpc^-1] / [Hz^-1]'''
    return 2*pi / dL_df(z) 

def X2Y(z):
    '''[Mpc^3] / [str * Hz] scalar conversion between observing and cosmological coordinates'''
    return dL_dth(z)**2 * dL_df(z)


# Filing functions

def strip_extension(filepath):
    if '.' not in filepath:
        return filepath, ''
    l = filepath.split('.')
    return ".".join(l[:-1]), '.'+l[-1]

def check_file_exists_and_increment(filepath):
    """
        Given filepath (path + filename), check if it exists. If so, add a _1
        at the end. etc.
    """
    if os.path.exists(filepath):
        filepath, ext = strip_extension(filepath)
        filepath += "_0" + ext
    else:
        return filepath
    n = 1
    while os.path.exists(filepath):
        filepath, ext = strip_extension(filepath)
        filepath = filepath[:-2] + "_" + str(n) + ext
        n += 1
    return filepath

def save_results(save_dict, data_cfg, out_cfg):
    version = data_cfg['root'].split('/')[-1]
    if 'version' in out_cfg: version = out_cfg['version']
    version += '_'+str(out_cfg['Nbls'])+"Nbls"
    if out_cfg['average_baselines']: version += '-avg'
    if out_cfg['random_vis']: version += "_random_vis"
    if out_cfg['zero']: version += "_zero-means"
    if out_cfg['part']: version += "_part_avg"
    if out_cfg['Nboots']: version += "_Nboot"+str(out_cfg['Nboots'])
    if out_cfg['covar']: version = "covar_" + version
    if 'chan_range' in data_cfg: version += "_ch" + data_cfg['chan_range']
    if 'outdir' in out_cfg: version = os.path.join(out_cfg['outdir'], version)
    if 'overwrite' in out_cfg: overwrite= bool(out_cfg['overwrite'])
    else: overwrite = False
    if overwrite:
        np.savez(version+".npz", **save_dict ) 
    else:
        np.savez(check_file_exists_and_increment(version+".npz"), **save_dict)
    sys.exit() 
