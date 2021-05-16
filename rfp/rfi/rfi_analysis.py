# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 the HERA Collaboration
# Licensed under the 2-clause BSD license.

import copy
import warnings

import numpy as np
from hera_qm import xrfi

def isolate_rfi(uvd, inplace=True, **filter_kwargs):
    """
    Use xrfi.medminfilt to isolate RFI.

    Parameters
    ----------
    uvd : UVData
        UVData object on which to isolate RFI. Assumes that this has 
        already been downselected to keep only the data on which to 
        perform the RFI analysis.

    inplace : bool, optional
        Whether to perform the RFI isolation on ``uvd`` or a copy 
        thereof. Default is to not make a copy.

    **filter_kwargs
        Keyword arguments to pass to xrfi.medminfilt.

    Returns
    -------
    rfi_uvd : UVData
        UVData object whose data array has been adjusted so that any 
        non-RFI signal has been removed (at least approximately).
    """
    # initialize a new data array
    rfi_data = np.zeros_like(uvd.data_array, dtype=uvd.data_array.dtype)

    # get baselines and polarizations
    baselines = np.unique(uvd.baseline_array)
    pols = uvd.get_pols()

    for baseline in baselines:
        for pol in pols:
            antpairpol = uvd.baseline_to_antnums(baseline) + (pol,)
            blt_inds, conj_blt_inds, pol_inds = uvd._key2inds(antpairpol)

            this_data = uvd.get_data(antpairpol)
            filt_data = xrfi.medminfilt(this_data, **filter_kwargs)
            this_rfi = this_data - filt_data

            rfi_data[blt_inds, 0, :, pol_inds[0]] = this_rfi
            if conj_blt_inds.size != 0:
                rfi_data[conj_blt_inds, 0, :, pol_inds[1]] = this_rfi.conj()

    if not inplace:
        rfi_uvd = copy.deepcopy(uvd)
        rfi_uvd.data_array = rfi_data
        return rfi_uvd

    uvd.data_array = rfi_data
    return uvd

def isolate_rfi_stations(vis_uvd, rfi_uvd=None, detrend='medfilt', 
                         apply_ws=True, detrend_nsig=100, 
                         ws_nsig=20, **detrend_kwargs):
    """
    Extract visibilities for just the narrowband transmitters.

    Parameters
    ----------
    vis_uvd : UVData
        UVData object containing the original visibility data. Assumes 
        downselection has already been applied.

    rfi_uvd : UVData, optional
        UVData object containing the RFI-only visibility data. If not 
        provided, then it is calculated from ``vis_uvd``.

    detrend : str or callable, optional
        String specifying which xrfi detrending method to use. May also 
        be a callable object that takes an array and kwargs and returns 
        an identically-shaped array whose entries give a modified 
        z-score (modified such that it is in units of standard deviations 
        when the input data is Gaussian). Default is to use 
        xrfi.detrend_medfilt.

    apply_ws : bool, optional
        Whether to apply a watershed algorithm to the detrended data. 
        Default is to apply watershed flagging.

    detrend_nsig : float, optional
        Lower bound for flagging data (i.e. any detrended data points 
        with values in excess of ``detrend_nsig`` will be flagged as 
        RFI). Default is 100 standard deviations (most narrowband 
        transmitters are *extremely* bright).

    ws_nsig : float, optional
        Lower bound for expanding RFI flags with the watershed algorithm. 
        Default is 20 standard deviations.

    **detrend_kwargs
        Keyword arguments passed directly to the detrending method used.

    Returns
    -------
    rfi_station_uvd : UVData
        UVData object whose data and flag arrays have been updated to 
        only contain information for narrowband transmitters, as 
        identified by this algorithm.
    """
    # get the detrending method to be used
    if callable(detrend):
        pass
    elif isinstance(detrend, str):
        try:
            detrend = getattr(xrfi, "detrend_%s" % detrend)
        except AttributeError:
            warnings.warn(
                "Detrending method not found. Defaulting to detrend_medfilt."
            )
            detrend = xrfi.detrend_medfilt
    else:
        raise TypeError("Unsupported type for ``detrend``.")

    # make rfi_uvd object if it doesn't exist
    if rfi_uvd is None:
        rfi_uvd = isolate_rfi(vis_uvd, inplace=False)

    # initialize new data and flag arrays
    new_rfi_data = np.zeros_like(
        rfi_uvd.data_array, dtype=rfi_uvd.data_array.dtype
    )
    new_rfi_flags = copy.deepcopy(rfi_uvd.flag_array)

    # prepare things to loop over
    baselines = np.unique(vis_uvd.baseline_array)
    pols = vis_uvd.get_pols()

    # now do things
    for baseline in baselines:
        for pol in pols:
            # get the (ant1, ant2, pol) key and data array indices
            antpairpol = vis_uvd.baseline_to_antnums(baseline) + (pol,)
            blt_inds, conj_blt_inds, pol_inds = vis_uvd._key2inds(antpairpol)

            # actually pull the data
            vis_data = vis_uvd.get_data(antpairpol)
            rfi_data = rfi_uvd.get_data(antpairpol)

            # figure out which pixels to flag
            vis_detrended = detrend(vis_data, **detrend_kwargs)
            these_flags = np.where(
                vis_detrended > detrend_nsig, True, False
            )

            if apply_ws:
                these_flags = xrfi._ws_flag_waterfall(
                    vis_detrended, these_flags, nsig=ws_nsig
                )

            # only use data in flagged channels
            this_rfi = np.where(these_flags, rfi_data, 0)

            # update the new arrays
            new_rfi_data[blt_inds, 0, :, pol_inds[0]] = this_rfi
            new_rfi_flags[blt_inds, 0, :, pol_inds[0]] = these_flags

            if conj_blt_inds.size != 0:
                new_rfi_data[conj_blt_inds, 0, :, pol_inds[1]] = this_rfi.conj()
                new_rfi_flags[conj_blt_inds, 0, :, pol_inds[1]] = these_flags

    # actually write the data and flags to a new UVData object
    rfi_station_uvd = copy.deepcopy(rfi_uvd)
    rfi_station_uvd.data_array = new_rfi_data
    rfi_station_uvd.flag_array = new_rfi_flags
    return rfi_station_uvd

def characterize_rfi_stations(rfi_station_uvd, duty_cycle_cut=0.1):
    """
    Take an RFI Station UVData object and characterize the transmitters.

    Parameters
    ----------
    rfi_station_uvd : UVData
        UVData object containing the RFI data to be characterized.

    duty_cycle_cut : float, optional
        Value between 0 and 1 designating the fraction of time a channel 
        must be flagged in order to identify a narrowband transmitter at 
        that channel's frequency. Default setting is 0.1.

    Returns
    -------
    rfi_station_parameters : dict
        Dictionary containing a summary of the RFI station parameters. 
        Currently, the parameters reported are as follows:
            transmitter_frequencies : center broadcasting frequency of 
                                      each transmitter
            transmitter_bandwidths : approximate bandwidth of each transmitter
            duty_cycles : fraction of time each transmitter is on
            signal_strengths : amplitude of transmitter visibility
            signal_strength_stds : standard deviation of each signal strength

        Frequency units are in Hz, visibility units are whatever units are 
        used by rfi_station_uvd.
    """
    # pull some useful metadata
    baselines = np.unique(rfi_station_uvd.baseline_array)
    pols = rfi_station_uvd.get_pols()
    freqs = rfi_station_uvd.freq_array.flatten()
    df = np.median(np.diff(freqs))

    rfi_info = {}

    for baseline in baselines:
        for pol in pols:
            # setup
            baseline = rfi_station_uvd.baseline_to_antnums(baseline)
            antpairpol = baseline + (pol,)
            if pol not in rfi_info:
                rfi_info[pol] = {}
            if baseline not in rfi_info[pol]:
                rfi_info[pol][baseline] = {}

            this_rfi = rfi_station_uvd.get_data(antpairpol)
            flags = rfi_station_uvd.get_flags(antpairpol)

            transmitter_freqs, transmitter_widths = find_transmitters(
                flags, freqs, duty_cycle_cut
            )

            duty_cycles = []
            sig_strengths = []
            strength_stds = []
            for freq, width in zip(transmitter_freqs, transmitter_widths):
                freq_in_freqs = [fq == freq for fq in freqs]
                if any(freq_in_freqs):
                    transmit_chans = freq_in_freqs.index(True)
                else:
                    transmit_chans = np.argwhere(
                        np.abs(freqs - freq) <= width
                    ).flatten()

                transmit_slice = (slice(None), transmit_chans)
                transmit_rfi = this_rfi[transmit_slice]
                transmit_flags = flags[transmit_slice]

                duty_cycle = np.mean(transmit_flags, axis=0)
                sig_strength = np.amax(np.abs(transmit_rfi), axis=0)
                strength_std = np.std(
                    np.abs(transmit_rfi[transmit_flags]), axis=0
                )

                duty_cycles.append(np.mean(duty_cycle))
                sig_strengths.append(np.mean(sig_strength))
                strength_stds.append(np.mean(strength_std))

            rfi_info[pol][baseline]["transmitter_frequencies"] = transmitter_freqs
            rfi_info[pol][baseline]["transmitter_bandwidths"] = transmitter_widths
            rfi_info[pol][baseline]["duty_cycles"] = duty_cycles
            rfi_info[pol][baseline]["signal_strengths"] = sig_strengths
            rfi_info[pol][baseline]["signal_strength_stds"] = sig_strength_stds
            
    return rfi_info

# TODO: add a function for further reducing the rfi_info dict from above func
# i.e. give a summary over all pols/baselines

def find_transmitters(flags, freqs, duty_cycle_cut):
    """
    Use a flag array to identify transmitter frequencies/bandwidths.

    Parameters
    ----------
    flags : np.ndarray
        Flag array on which to perform the analysis.

    freqs : array-like
        Frequency array corresponding to the flag array's frequency axis.

    duty_cycle_cut : float
        Number between 0 and 1 specifying the minimum fraction of time a 
        channel must be flagged to be considered to contain a transmitter.

    Returns
    -------
    transmitter_freqs : list
        List of the estimated center frequency for each narrowband transmitter.

    transmitter_widths : list
        List of the estimated bandwidth for each narrowband transmitter.
    """
    freqs = np.array(freqs).flatten()
    df = np.median(np.diff(freqs))

    duty_cycles = np.mean(flags, axis=0)
    transmit_chans = np.argwhere(duty_cycles > duty_cycle_cut).flatten()
    transmitter_freqs = []
    transmitter_widths = []
    for ch_ind, chan in enumerate(transmitter_chans):
        before, after = check_nearby_channels(chan, transmit_chans)
        if not (before or after):
            # transmitter contained to a single frequency channel
            transmitter_freqs.append(freqs[chan])
            transmitter_widths.append(df)
            continue

        min_chan, max_chan = chan, chan
        min_ind, max_ind = ch_ind, ch_ind

        while before:
            min_ind -= 1
            min_chan = transmit_chans[min_ind]
            before = check_nearby_channels(min_chan, transmit_chans)[0]

        while after:
            max_ind += 1
            max_chan = transmit_chans[max_ind]
            after = check_nearby_channels(max_chan, transmit_chans)[1]

        trans_freq = 0.5 * (freqs[min_chan] + freqs[max_chan])
        trans_bw = freqs[max_chan] - freqs[min_chan]
        if trans_freq not in transmitter_freqs:
            transmitter_freqs.append(trans_freq)
            transmitter_widths.append(trans_bw)

    return transmitter_freqs, transmitter_widths

def check_nearby_channels(chan, chans):
    """
    Determine if any channels in ``chans`` are adjacent to ``chan``.

    This is merely a helper function for the ``find_transmitters`` function.

    Parameters
    ----------
    chan : int
        Channel of interest.

    chans : array-like of int
        Array of channel numbers, sorted in increasing order.

    Returns
    -------
    before : bool
        Whether there is a channel directly preceding ``chan`` in ``chans``.

    after : bool
        Whether there is a channel directly following ``chan`` in ``chans``.
    """
    ch_ind = list(chans).index(chan)
    Nchan = len(chans)
    
    after = False if ch_ind == Nchan - 1 else chan + 1 == chans[ch_ind + 1]
    before = False if ch_ind == 0 else chan - 1 == chans[ch_ind - 1]
    return before, after
