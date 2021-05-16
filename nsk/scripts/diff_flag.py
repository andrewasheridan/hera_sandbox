# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import numpy as np
from scipy import signal
from six.moves import map, range
from hera_cal.datacontainer import DataContainer


def diff_norm(data, axis=0, pad=20, ks=25):
    """
    Difference data and normalize by median

    Args:
        data : complex ndarray or DataContainer
            visibility waterfall of shape (Ntimes, Nfreqs)
            or DataContainer of waterfalls
        axis : int or tuple of int
            Axis of data along which to take difference.
            Will iterate if axis is fed as a tuple.
        pad : int
            Reflect the data along freq axis by "pad" number
            of pixels before taking median filter.
            Used to limit edge effects in median filter.
        ks : int
            Kernel size of median filter (in freq channels)
            applied to the time-averaged spectrum that is used
            to normalize the data.

    Returns:
        ndarray or DataContainer
            Differenced data normalized by median
    """
    if isinstance(data, hc.datacontainer.DataContainer):
        diff = hc.datacontainer.DataContainer({})
        for k in data:
            diff[k] = diff_norm(data[k], axis=axis, pad=pad, ks=ks)
        return diff

    # parse inputs
    if isinstance(axis, (int, np.int)):
        axis = (axis,)

    # setup diff
    diff = data.copy()

    # iterate over axis
    for ax in axis:
        d1 = np.roll(diff, 1, axis=ax) - diff
        d2 = np.roll(diff, -1, axis=ax) - diff
        diff = np.min([np.abs(d1), np.abs(d2)], axis=0)

    # take median along axis to detrend and median filter it
    m = np.median(diff, axis=0)
    ms = signal.medfilt(np.pad(m, pad, 'reflect', reflect_type='odd'), kernel_size=ks)
    if pad > 0:
        ms = ms[pad:-pad]

    # divide diff by trend
    ms[np.isclose(ms, 0.0)] = 1
    diff /= ms

    return diff

def diff_flag(diff, nsig=2.5, lowcut=1e-1):
    """
    Flag differenced data at a specified tolerance.

    Args:
        diff : ndarray or DataContainer
            Diffed and normed data to flag. If fed as
            a DataContainer, takes the mean of its values first.
        nsig : float
            Amplitude above which diffed data are flagged.
            Diffed data are assumed to be normalized.
        lowcut : float
            Flag diffed data below this amplitude.

    """
    if isinstance(diff, hc.datacontainer.DataContainer):
        diff = np.mean(list(diff.values()), axis=0)
        return diff_flag(diff, nsig=nsig, lowcut=lowcut)
    return (diff < lowcut) | (diff > nsig)

def diff_thresh(diff, time_sig=None, time_ks=21, time_pad=0, edgecut=0, freq_sig=None, freq_ks=21, freq_pad=0):
    """
    Broadcast (or threshold) differencd data and flag

    Args:
        diff : ndarray or DataContainer
            Differenced and normalized visibility data.
            Will take mean of values if fed as DataContainer
        time_sig : float
            Nsigma threshold for flagging time integrations after
            collapsing data across frequency axis. Default (None) is no time thresholding.
        time_ks : int
            median filter kernel size for smoothing across time after
            collapsing data across frequency.
        time_pad : int
            Padding factor along time before taking median filter
        edgecut : int
            Number of channels to exclude on either side when collapsing
            data across frequency axis.
        freq_sig : Nsigma threshold for flagging freq bins after
            collapsing data across time axis. Default (None) is no freq thresholding.
        freq_ks : int
            median filter kernel size for smoothing across freq
            after collaping across time.
        freq_pad : int
            padding factor along freq before taking median filter

    Returns:
        ndarray
            Boolean flag array after thresholding
    """
    if isinstance(diff, DataContainer):
        return diff_thresh(np.mean(list(diff.values()), axis=0), time_sig=time_sig, time_ks=time_ks,
                           time_pad=time_pad, edgecut=edgecut, freq_sig=freq_sig, freq_ks=freq_ks,
                           freq_pad=freq_pad)

    # time threshold
    tflags = np.zeros((diff.shape[0], 1), np.bool)
    if time_sig is not None:
        if isinstance(edgecut, (int, np.int)):
            edgecut = [edgecut, edgecut]
        edgecut = list(edgecut)
        edgecut = slice(edgecut[0], diff.shape[1] - edgecut[1])
        tm = np.median(diff[:, edgecut], axis=1)
        tms = signal.medfilt(np.pad(tm, time_pad, 'reflect', reflect_type='odd'), kernel_size=time_ks)
        if time_pad > 0:
            tms = tms[time_pad:-time_pad]
        tm = tm / tms - 1
        tmad = np.nanmedian(np.abs(tm - np.nanmedian(tm))) * 1.48
        tflags = (np.abs(tm) > tmad * time_sig)[:, None]
     
    # freq threshold
    fflags = np.zeros((1, diff.shape[1]), np.bool)
    if freq_sig is not None:
        fm = np.median(diff, axis=0)
        fms = signal.medfilt(np.pad(fm, freq_pad, 'reflect', reflect_type='odd'), kernel_size=freq_ks)
        if freq_pad > 0:
            fms = fms[freq_pad:-freq_pad]
        fm = fm / fms - 1
        fmad = np.nanmedian(np.abs(fm - np.nanmedian(fm))) * 1.48
        fflags = (np.abs(fm) > fmad * freq_sig)[None, :]
    
    # join flags
    return tflags | fflags

