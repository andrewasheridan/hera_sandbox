"""
This script plots histograms and waterfalls for all of the 
sum files that have an associated detrended file. This is a 
workaround to the recurring problem of Jupyter throwing a 
MemoryError after some number of iterations (even though a 
MemoryError *should not* ever be thrown...).

This is very messy, as it is mostly a copy-pasta of the code 
in the rfi_exploration notebook.
"""

import calendar
import copy
import datetime
import glob
import itertools
import os
import re
import sys
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy import units
from astropy.time import Time
from astropy.coordinates import EarthLocation

import uvtools
from uvtools import dspec
from uvtools.plot import waterfall
from pyuvdata import UVData
from hera_mc import cm_hookup
from hera_qm import xrfi

import hera_sim
if hera_sim.version.version.startswith('0'):
    from hera_sim.rfi import _listify
else:
    from hera_sim.utils import _listify

# Define some helper functions.

def get_ant_to_node_map():
    """Get a dictionary mapping antenna numbers to node numbers."""
    
    # load in the hookup information for HERA, as of now
    db_hookup = cm_hookup.Hookup()
    
    # take only the antenna/node information from the hookup table
    hookup_dict = db_hookup.get_hookup("HH")
    ant_to_node_table = db_hookup.show_hookup(
        hookup_dict, cols_to_show=("antenna", "node")
    )
    
    # use regular expressions to pull the antenna/node numbers
    ant_pattern = re.compile("A[0-9]+")
    node_pattern = re.compile("N[0-9]+")
    ants = ant_pattern.findall(ant_to_node_table)
    nodes = node_pattern.findall(ant_to_node_table)

    # convert the antennas/nodes to integers (for simplifying their use)
    ants = [int(ant[1:]) for ant in ants] # remove the leading A
    nodes = [int(node[1:]) for node in nodes] # remove the leading N
    
    # return the ant : node mapping
    return dict(zip(ants, nodes))

def sort_by_node(data, antennas, node_to_ant_map):
    """Sort a data array by node number.
    
    Assumes that the zeroth axis of the data runs along antennas.
    """
    if not isinstance(antennas, list):
        antennas = list(antennas)
    # be aware of using log scale or not
    sorted_data = np.zeros_like(data) if data.min() == 0 else np.ones_like(data)
    rownum = 0
    for ants in node_to_ant_map.values():
        for ant in ants:
            if ant not in antennas: continue
            ant_ind = antennas.index(ant)
            sorted_data[rownum] = data[ant_ind]
            rownum += 1
    return sorted_data

def get_data_like_cube(uvd, attr='data'):
    Nants = uvd.Nants_data
    Ntimes = uvd.Ntimes
    Nfreqs = uvd.Nfreqs
    Npols = uvd.Npols
    data_cube = np.zeros(
        (Nants, Ntimes, Nfreqs, Npols), 
        dtype=np.float
    )

    for i, antpair in enumerate(uvd.get_antpairs()):
        for j, pol in enumerate(uvd.get_pols()):
            antpairpol = antpair + (pol,)
            data = getattr(uvd, f"get_{attr}")(antpairpol)
            data_cube[i, :, :, j] = data
        
    return data_cube

def get_transmitter_channels(flags, threshold=0.9):
    # flags shape=(Nants, Ntimes, Nfreqs, Npols)
    transmitter_likelihoods = flags.mean(axis=0).mean(axis=0)
    has_transmitter = np.where(
        transmitter_likelihoods > threshold, True, False
    )
    # logical AND over ants; logical OR over pols
    transmitter_channels = np.argwhere(
        np.sum(has_transmitter, axis=-1).astype(bool)
    )
    return transmitter_channels.flatten()

def find_transmitter_broadcasting_params(transmitter_channels, freqs):
    transmitter_freqs = []
    df = np.median(np.diff(freqs))
    transmitter_bandwidths = []
    processed_channels = []
    for index, channel in enumerate(transmitter_channels):
        if channel in processed_channels:
            continue
        processed_channels.append(channel)
        end_index = index
        end_channel = channel
        try:
            while transmitter_channels[end_index + 1] == end_channel + 1:
                end_index += 1
                end_channel += 1
                processed_channels.append(end_channel)
        except IndexError:
            pass
        transmitter_freqs.append(0.5 * (freqs[channel] + freqs[end_channel]))
        transmitter_bandwidths.append(max(df, freqs[end_channel] - freqs[channel]))
    return transmitter_freqs, transmitter_bandwidths

def extract_datetime_times(uvd):
    """
    Extract the times from a UVData object in datetime format.
    """
    times = np.unique(uvd.time_array)
    loc = EarthLocation(*uvd.telescope_location_lat_lon_alt)
    times = Time(times, format='jd', location=loc).to_datetime()
    return times

def convert_to_local_times(times):
    """
    Take datetime objects and return local times in seconds relative to midnight.
    """
    ref_day = times[0].day if times[0].hour < 12 else times[0].day + 1
    times = np.array([
        datetime.timedelta(
            t.day - ref_day, seconds=t.second, 
            minutes=t.minute, hours=t.hour
        ).total_seconds() for t in times
    ]).flatten()
    return times

def extract_local_times(uvd):
    """
    Pull the times from a UVData object in local time (seconds from midnight).
    """
    times = extract_datetime_times(uvd)
    return convert_to_local_times(times)

def construct_plot_grid(Nrows, Ncols, figsize=None, dpi=100, widths=None, heights=None):
    """
    Construct an Nrows x Ncols grid of plots on a single figure, like a calendar.
    
    x- and y-axes are hidden for every plot.
    
    figsize : tuple, size of figure
    widths : list of float, length Ncols, must sum to 1
    heights: list of float, length Nrows, must sum to 1
    
    Returns
    -------
    fig : Figure object
    axes_dict : dict mapping matrix-style location (i,j) to Axes objects
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes_dict = {}
    if widths is None:
        widths = [1 / Ncols for col in range(Ncols)]
    if heights is None:
        heights = [1 / Nrows for row in range(Nrows)]
        
    if not np.isclose(sum(widths), 1):
        raise ValueError("Sum of widths must be unity.")
    if not np.isclose(sum(heights), 1):
        raise ValueError("Sum of heights must be unity.")
    if len(widths) != Ncols:
        raise ValueError("Widths must have same length as number of columns.")
    if len(heights) != Nrows:
        raise ValueError("Heights must have same length as number of rows.")
        
    for col, width in enumerate(widths):
        for row, height in enumerate(heights):
            axes_id = (row, col)
            axes_xy = (col * width, 1 - row * height)
            ax = fig.add_axes(axes_xy + (width, height))
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            axes_dict[axes_id] = ax
    return fig, axes_dict

# Setup

data_path = "/lustre/aoc/projects/hera/rpascua/rfi/narrowband_data"
dfiles = sorted(glob.glob(os.path.join(data_path, "*")))
detrended_files = list(dfile for dfile in dfiles if 'detrended' in dfile)
sum_files = list(dfile for dfile in dfiles if 'sum' in dfile)
diff_files = list(dfile for dfile in dfiles if 'diff' in dfile)
jd_pattern = re.compile("[0-9]{7}")
sum_files_to_use = [
    f for f in sum_files 
    if os.path.exists(f.replace('autos.sum','detrended.flagged'))
]
detrended_files_to_use = [
    f.replace('autos.sum', 'detrended.flagged') for f in sum_files_to_use
]

save_path = '/lustre/aoc/projects/hera/rpascua/rfi/narrowband_data/plots'
worst_offenders_by_jd = {} # make sure to write these to disk!!!
group_size = 10
offender_id_thresh = 0.3
transmitter_thresh = 0.5
ant_to_node_map = get_ant_to_node_map()
dpi = 300
Nticks = 6
spine_lw = 3

for sum_file, det_file in zip(sum_files_to_use, detrended_files_to_use):
    jd = re.compile('[0-9]{7}').findall(sum_file)[0]
    uvd = UVData()
    uvd.read(sum_file)
    data = get_data_like_cube(uvd)
    del uvd
    uvd = UVData()
    uvd.read(det_file)
    flags = get_data_like_cube(uvd, 'flags')
    
    # Extract some useful metadata.
    Nants, Ntimes, Nfreqs, Npols = data.shape
    times = extract_local_times(uvd)
    freqs_MHz = np.unique(uvd.freq_array) / 1e6
    ants = uvd.antenna_numbers
    pols = uvd.get_pols()
    linpols = [pol for pol in pols if pol[0] == pol[1]]
    crosspols = [pol for pol in pols if pol[0] != pol[1]]
    lin_inds = [pols.index(pol) for pol in linpols]
    cross_inds = [pols.index(pol) for pol in crosspols]
    
    # Find the transmitters
    thresh = 0.5
    transmitter_chans = get_transmitter_channels(flags, transmitter_thresh)
    transmitter_freqs, transmitter_bws = find_transmitter_broadcasting_params(
        transmitter_chans, freqs_MHz
    )
    if len(transmitter_chans) == 0:
        print(f"No transmitters found for JD{jd}.")
        continue
    # look at some histograms for transmitters
    linpol_histograms = {}
    crosspol_histograms = {}
    lin_bounds = (data[...,lin_inds].min(), data[...,lin_inds].max())
    cross_bounds = (data[...,cross_inds].min(), data[...,cross_inds].max())
    Nbins = 1000
    lin_bin_edges = np.linspace(*lin_bounds, Nbins + 1)
    lin_bin_centers = 0.5 * (lin_bin_edges[1:] + lin_bin_edges[:-1])
    cross_bin_edges = np.linspace(*cross_bounds, Nbins + 1)
    cross_bin_centers = 0.5 * (cross_bin_edges[1:] + cross_bin_edges[:-1])
    
    # Calculate reference data.
    full_lin_hist = np.histogram(
        data[...,lin_inds][~flags[...,lin_inds].astype(bool)], 
        lin_bin_edges, density=True
    )[0]
    full_cross_hist = np.histogram(
        data[...,cross_inds][~flags[...,cross_inds].astype(bool)], 
        cross_bin_edges, density=True
    )[0]
    ymin = 0
    ymax = full_lin_hist.max() * 1.1
    yticks = np.linspace(ymin, ymax, 3, endpoint=False)
    lin_median = np.median(data[...,lin_inds][~flags[...,lin_inds].astype(bool)])
    cross_median = np.median(data[...,cross_inds][~flags[...,cross_inds].astype(bool)])
    medians = [lin_median, cross_median]

    fig, axes_dict = construct_plot_grid(
        len(transmitter_freqs), 2, figsize=(15,50), dpi=dpi
    )
    for row, freq in enumerate(transmitter_freqs):
        chan = np.argmin(np.abs(freqs_MHz - freq))
        lin_autos = data[:,:,chan,lin_inds].flatten()
        lin_hist = np.histogram(lin_autos, lin_bin_edges, density=True)[0]
        cross_autos = data[:,:,chan,cross_inds].flatten()
        cross_hist = np.histogram(cross_autos, cross_bin_edges, density=True)[0]
        hist_iter = zip(
            (lin_hist, cross_hist), (full_lin_hist, full_cross_hist), 
#            (clean_lin_hist, clean_cross_hist),
            (lin_bin_centers, cross_bin_centers),
        )
        for col, hists_and_bins in enumerate(hist_iter):
            hist, full_hist, bins = hists_and_bins
            ax = axes_dict[(row,col)]
            plt.setp(ax.spines.values(), color='darkorchid', lw=3)
            if col == 0:
                title = "Linear Polarizations"
                ax.set_ylabel("Density", fontsize=12)
                ax.yaxis.set_visible(True)
            else:
                title = "Cross Polarizations"

            if row == 0:
                ax.set_title(title, fontsize=12)
            if row == len(transmitter_freqs) - 1:
                ax.set_xlabel("Autocorrelation Power [arb]", fontsize=12)
                ax.xaxis.set_visible(True)
            ax.plot(bins, hist, color='red', alpha=0.7, label='Transmitter')
            ax.plot(
                bins, full_hist, color='dodgerblue', alpha=0.7, 
                label='Unflagged Samples'
            )
            ax.axvline(medians[col], color='k', ls='--', label='Unflagged Median')
            ax.legend(ncol=1, loc='center right')
            ax.set_ylim(ymin, ymax)
            ax.set_yticks(yticks)
            
            ax.text(
                0.97, 0.97, f'{freq:.2f} MHz', transform=ax.transAxes,
                va='top', ha='right'
            )
            
    filename = os.path.join(save_path, f"{jd}_histograms.pdf")
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    fig.savefig(filename.replace('.pdf', '.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # Setup for finding worst offenders
    channel_complement = np.ones(freqs_MHz.size, dtype=bool)
    channel_complement[transmitter_chans] = False
    not_trans_slice = (slice(None), slice(None), channel_complement, slice(None))
    median_autos = np.median(data[not_trans_slice], axis=(1,2))
    abs_deviations = np.abs(
        data - median_autos[:,None,None,:] * np.ones(data.shape)
    ) / median_autos[:,None,None,:]

    # Actually find the worst offenders
    worst_offenders = np.empty((data.shape[0], data.shape[-1], group_size))
    for ant_ind in range(data.shape[0]):
        for pol_ind in range(data.shape[-1]):
            max_deviations = np.max(abs_deviations[ant_ind,:,:,pol_ind], axis=0)
            worst_offenders[ant_ind, pol_ind] = freqs_MHz[
                np.argsort(max_deviations)[::-1]
            ][:group_size]
            
    worst_offender_occurrences = dict.fromkeys(transmitter_freqs, 0)
    for ant_ind in range(Nants):
        for pol_ind in range(Npols):
            offenders = worst_offenders[ant_ind, pol_ind]
            inds = list(set(
                np.argmin(np.abs(fq - transmitter_freqs)) for fq in offenders
            ))
            for fq in np.array(transmitter_freqs)[inds]:
                worst_offender_occurrences[fq] += 1
                
    Ngroups = Nants * Npols
    worst_offender_freqs = np.array(transmitter_freqs)[
        np.array(
            list(worst_offender_occurrences.values())
        ) / Ngroups > offender_id_thresh
    ]
    
    # Finally, track the result
    worst_offenders_by_jd[jd] = worst_offender_freqs
    
    # We're not done yet--gotta also make the plots!
    # Get antenna/node mapping for plot formatting
    node_to_ant_map = {}
    for ant in ants:
        node = ant_to_node_map[ant]
        if node not in node_to_ant_map.keys():
            node_to_ant_map[node] = []
        node_to_ant_map[node].append(ant)
    sorted_ants = np.concatenate([ants for ants in node_to_ant_map.values()])
    
    # Convert data to log-scale and mask data that didn't like the conversion
    use_data = np.log10(sort_by_node(data, ants, node_to_ant_map))
    use_data = np.ma.MaskedArray(
        use_data, np.logical_or(np.isinf(use_data), np.isnan(use_data))
    )
    # Ugly copypasta
    fig, axes_dict = construct_plot_grid(Nants, Npols, dpi=dpi, figsize=(20,50))
    vmin = 5
    vmax = use_data.max()
    norm = plt.cm.colors.Normalize(vmin, vmax)
    cmap = plt.cm.plasma
    smap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    extent = (freqs_MHz.min(), freqs_MHz.max(), times.max(), times.min())
    yticks = np.linspace(-6, 4, Nticks) * units.hr.to('s')
    yticklabels = [r'%d$^h$' % (hr % 24) for hr in np.linspace(-6,4,Nticks)]
    for row in range(Nants):
        for col in range(Npols):
            ax = axes_dict[(row,col)]
            plt.setp(ax.spines.values(), lw=spine_lw, color='gold')
            ax.imshow(
                use_data[row,:,:,col], aspect='auto', 
                cmap=cmap, norm=norm, extent=extent
            )
            ax.set_facecolor('k')

            if row == 0:
                # Set the column title
                title = f"{pols[col]} Polarization"
                ax.set_title(title, fontsize=12)

            if col == 0:
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
                ax.set_ylabel("Local Time", fontsize=12)
                ax.yaxis.set_visible(True)

            if row == Nants - 1:
                ax.set_xlabel("Frequency [MHz]", fontsize=12)
                ax.xaxis.set_visible(True)

            if col == Npols - 1:
                ant = sorted_ants[row]
                node = ant_to_node_map[ant]
                twin_ax = ax.twinx()
                plt.setp(twin_ax.spines.values(), lw=0)
                twin_ax.xaxis.set_visible(False)
                twin_ax.yaxis.set_visible(True)
                twin_ax.tick_params('y', right=False)
                twin_ax.set_yticklabels([])
                twin_ax.set_ylabel(f"N{node}A{ant}", fontsize=12, rotation=270, labelpad=5)

            ax.set_ylim(
                yticks[-1] + 1 * units.hr.to('s'), 
                yticks[0] - 1 * units.hr.to('s')
            )

            tmin, tmax = ax.get_ylim()
            y1 = (times.min() - tmin) / (tmax - tmin)
            y2 = (times.max() - tmin) / (tmax - tmin)
            for transmitter_freq in transmitter_freqs:
                is_worst_offender = transmitter_freq in worst_offender_freqs
                color = 'yellow' if is_worst_offender else 'white'
                ax.axvline(transmitter_freq, 0, y2, color=color, lw=0.5)
                ax.axvline(transmitter_freq, y1, 1, color=color, lw=0.5)

    cbar = fig.colorbar(
        mappable=smap, ax=[ax for ax in axes_dict.values()], 
        orientation='vertical', pad=0.02, aspect=50,
        extend='min'
    )
    cbar.set_label(
        "Autocorrelation Power [arb. units] (log10)", 
        fontsize=12, rotation=270
    )
    
    # Finally, save the picture and close the plot
    save_filename = os.path.join(
        save_path, f"{jd}_waterfalls_with_transmitters_and_worst_offenders.pdf"
    )
    fig.savefig(save_filename, dpi=dpi, bbox_inches='tight')
    fig.savefig(save_filename.replace('.pdf','.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    del uvd

# Write the worst offenders to disk.
save_path = '/lustre/aoc/projects/hera/rpascua/rfi/narrowband_data'
save_filename = os.path.join(save_path, 'worst_offenders_info.npz')
np.savez(save_filename, **worst_offenders_by_jd)
