#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 UPennEoR
# Licensed under the 2-clause BSD License

from __future__ import print_function, division, absolute_import
from pyuvdata import UVData, UVCal
import numpy as np
import argparse
import os
import h5py
import hera_cal

ap = argparse.ArgumentParser(description="Convert CASA gain solutions in hdf5 files to .calfits files")

ap.add_argument("--fname", type=str, help="name of calfits file", required=True)
ap.add_argument("--uv_file", type=str, help="name of uvfile to apply calibration to", required=True)
ap.add_argument("--kcal", type=str, help="name of file containing delay solutions", required=True)
ap.add_argument("--bcal", type=str, help="name of file containing bandpass solutions", required=True)
ap.add_argument("--acal", type=str, help="name of file containing absolute scale reference spectrum", required=True)
ap.add_argument("--overwrite", default=False, action="store_true", help="overwrite output file if it exists")
ap.add_argument("--multiply_gains", default=False, action="store_true", help="change gain convention from divide to multiply")


def main(ap):
    # read in UVData object
    uv_file = args.uv_file
    uvd = UVData()
    uvd.read(uv_file)

    # get metadata
    antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=True)
    ants = list(ants)
    freqs = np.unique(uvd.freq_array[0, :])
    times = np.unique(uvd.time_array)
    Nants = len(ants)
    Nfreqs = len(freqs)
    Ntimes = len(times)
    # assume we have linear polarizations x and y for gain solutions, in that order
    jones_array = ['x', 'y']
    Njones = len(jones_array)

    # start with unity gains
    gains = np.ones((Nants, Nfreqs, 1, Njones), dtype=np.complex)
    flags = np.zeros((Nants, Nfreqs, 1, Njones), dtype=np.bool)

    # read in K-gain file
    kfile = args.kcal
    delays = np.zeros_like(ants, dtype=np.float)
    with h5py.File(kfile, 'r') as f:
        delay_ants = f["/Data/delay_ants"].value.tolist()
        delays = f["/Data/delays"].value.T
        delay_flags = f["/Data/delay_flags"].value.T
    # convert from ns -> s
    delays *= 1e-9

    # reorder antennas
    delays = np.array(map(lambda a: delays[delay_ants.index(a), :] if a in delay_ants else 0., ants))
    delay_flags = np.array(map(lambda a: delay_flags[delay_ants.index(a), :] if a in delay_ants else True, ants))

    # convert delays into complex gains; freqs has units of Hz
    delay_gains = np.exp(2j * np.pi * np.einsum('a,bc->bac', freqs, delays))[:, :, np.newaxis, :]
    delay_gain_flags = delay_flags[:, np.newaxis, :]
    delay_gain_flags = np.repeat(delay_gain_flags, Nfreqs, axis=1)[:, :, np.newaxis, :]

    # multiply into gains
    gains *= delay_gains

    # add flags
    flags += delay_gain_flags

    # read in B-gain file
    bfile = args.bcal
    with h5py.File(bfile, 'r') as f:
        bp_gains = np.swapaxes(f["/Data/bp"].value, 0, 2)
        bp_ants = f["/Data/bp_ants"].value.tolist()
        bp_freqs = f["/Data/bp_freqs"].value
        bp_flags = np.swapaxes(f["/Data/bp_flags"].value, 0, 2)

    # get number of frequencies
    bp_Nfreqs = len(bp_freqs)
    # reorder antennas
    bp_gains = np.array([bp_gains[bp_ants.index(a), :, :].squeeze() if a in bp_ants
                         else np.ones((bp_Nfreqs, Njones), dtype=np.complex) for a in ants])
    bp_flags = np.array([bp_flags[bp_ants.index(a), :, :].squeeze() if a in bp_ants
                         else np.ones((bp_Nfreqs, Njones), dtype=np.bool) for a in ants])

    # get gains and flags into the right shape
    bp_gains = bp_gains[:, :, np.newaxis, :]
    bp_flags = bp_flags[:, :, np.newaxis, :]

    # multipy into gains
    gains *= bp_gains

    # add flags
    flags += bp_flags

    # read in overall amplitude spectrum
    afile = args.acal
    with h5py.File(afile, 'r') as f:
        amp = f["/Data/spectrum_scale"].value

    # turn it into the right shape
    amp = np.stack((amp,) * Njones).T
    amp = np.stack((amp,) * Nants)
    right_shape = (Nants, Nfreqs, Njones)
    if amp.shape != (Nants, Nfreqs, Njones):
        raise ValueError("amp shape is {}; was expecting {}".format(amp.shape, right_shape))

    # add an extra dummy time axis
    amp = amp[:, :, np.newaxis, :]

    # multiply into the gains; we take the square root so that g_i * g_j^* gives the original
    gains *= np.sqrt(amp)

    # make the gains and flags the right number of time samples
    gains = np.repeat(gains, Ntimes, axis=2)
    flags = np.repeat(flags, Ntimes, axis=2)

    # make into calfits
    fname = args.fname
    overwrite = args.overwrite
    if args.multiply_gains:
        gain_convention = 'multiply'
    else:
        gain_convention = 'divide'
    gain_dict = {}
    flag_dict = {}
    for i, ant in enumerate(ants):
        for j, pol in enumerate(jones_array):
            gain_dict[(ant, pol)] = gains[i, :, :, j].T.conj()
            flag_dict[(ant, pol)] = flags[i, :, :, j].T
    uvc = hera_cal.io.write_cal(fname, gain_dict, freqs, times, flags=flag_dict,
                                overwrite=overwrite, gain_convention=gain_convention)

    return


if __name__ == '__main__':
    # parse args
    args = ap.parse_args()

    main(ap)
