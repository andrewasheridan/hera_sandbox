#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright (c) 2018 UPennEoR
# Licensed under the 2-clause BSD License
"""
This script is casa_cal_to_hdf5.py.

This script is designed to take delay and bandpass calibration solutions from CASA
tables and save them as hdf5 files. Then the accompanying hdf5_to_calfits.py program
will translate them to a calfits file suitable for applying with hera_cal.apply_cal.

This script is based on a script originally by Nick Kern, located at:
https://github.com/HERA-Team/hera_sandbox/blob/master/nsk/scripts/sky_image.py

This script should be called by the python interpreter that ships with CASA.
"""
from __future__ import print_function, division, absolute_import
import argparse
import h5py

ap = argparse.ArgumentParser(description="Run with casa as: casapy -c casa_cal_to_hdf5.py <args>")
ap.add_argument('--script', '-c', type=str, help='name of this script', required=True)
ap.add_argument('--Kcal', '-k', type=str, help='name of file where K calibration is saved', required=True)
ap.add_argument('--Bcal', '-b', type=str, help='name of file where B calibration is saved', required=True)

if __name__ == '__main__':
    # parse args
    args = ap.parse_args()

    # convert K-gain file to hdf5 file
    kcal = args.Kcal
    print("extracting gains from {}...".format(kcal))
    tb.open(kcal)
    delays = tb.getcol('FPARAM').squeeze()
    delay_ants = tb.getcol('ANTENNA1')
    delay_flags = tb.getcol('FLAG').squeeze()
    tb.close()

    # save out to hdf5
    fileout = "{}.h5".format(kcal)
    print("writing {}...".format(fileout))
    with h5py.File(fileout, 'w') as f:
        header = f.create_group("Header")
        header["Kcal_file"] = kcal

        dgrp = f.create_group("Data")
        dgrp["delay_ants"] = delay_ants
        dgrp["delays"] = delays
        dgrp["delay_flags"] = delay_flags


    # convert B-gain file to hdf5 file
    bcal = args.Bcal
    print("extracting bandpass from {}...".format(bcal))
    tb.open(bcal)
    bp = tb.getcol('CPARAM').squeeze()
    bp_ants = tb.getcol('ANTENNA1')
    bp_flags = tb.getcol('FLAG').squeeze()
    tb.close()
    # load spectral window info
    tb.open(bcal + "/SPECTRAL_WINDOW")
    bp_freqs = tb.getcol("CHAN_FREQ")
    tb.close()

    # save out to hdf5
    fileout = "{}.h5".format(bcal)
    print("saving {}...".format(fileout))
    with h5py.File(fileout, 'w') as f:
        header = f.create_group("Header")
        header["Bcal_file"] = bcal

        dgrp = f.create_group("Data")
        dgrp["bp"] = bp
        dgrp["bp_ants"] = bp_ants
        dgrp["bp_flags"] = bp_flags
        dgrp["bp_freqs"] = bp_freqs
