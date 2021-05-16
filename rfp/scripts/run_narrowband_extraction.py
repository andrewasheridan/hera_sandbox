# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 the HERA Collaboration
# Licensed under the 2-clause BSD license.
import os
import re
import sys

import numpy as np
from hera_qm import xrfi
from pyuvdata import UVData

# Some file setup stuff.
auto_file = sys.argv[1]
save_dir = sys.argv[2]
pattern = re.compile("[0-9]{7}")
jd = pattern.findall(auto_file)[0]
uvd = UVData()
uvd.read(auto_file)

# Do the detrending and flagging.
data_shape = uvd.data_array.shape
detrended_rfi_data = np.zeros(data_shape, dtype=np.complex)
rfi_flags = np.zeros(data_shape, dtype=np.bool)
# Flag aggressively
medfilt_nsig = 5
ws_nsig = 2

for antpairpol, data in uvd.antpairpol_iter():
    # get indices for properly slicing through data array
    blt_inds, conj_blt_inds, pol_inds = uvd._key2inds(antpairpol)
    if len(blt_inds) > 0:
        this_slice = (blt_inds, 0, slice(None), pol_inds[0])
    else:
        this_slice = (conj_blt_inds, 0, slice(None), pol_inds[1])

    # detrend the original data to find where the stations are
    detrended_data = xrfi.detrend_medfilt(data)
    station_flags = np.where(
        np.abs(detrended_data.real) > medfilt_nsig, True, False
    )

    # update flags with watershed algorithm
    station_flags = xrfi._ws_flag_waterfall(
        np.abs(detrended_data.real), station_flags, nsig=ws_nsig
    )
    rfi_flags[this_slice] = station_flags
    detrended_rfi_data[this_slice] = detrended_data

history = "\nFlagged narrowband RFI using median filter detrending and "
history += f"a watershed algorithm. Median filter cut is {medfilt_nsig} " 
history += f"sigma; watershed algorithm cut is {ws_nsig}."
uvd.history += history

# Just write the detrended data to disk with the flags.
save_filename = os.path.join(
    save_dir, f"{jd}.detrended.flagged.uvh5"
)
uvd.flag_array = rfi_flags
# XXX this is currently breaking bc file permissions. Figure out an alternative.
#uvd.write_uvh5(auto_file, clobber=True)
uvd.data_array = detrended_rfi_data
uvd.history += " Data array replaced with z-scores from median detrending."
uvd.write_uvh5(save_filename, clobber=True)
