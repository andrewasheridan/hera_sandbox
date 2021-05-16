# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 the HERA Collaboration
# Licensed under the 2-clause BSD license.

import argparse
import glob
import os
import re
import warnings
import yaml

from pyuvdata.utils import polstr2num
from pyuvdata import UVData

def construct_file_glob(obs_season=None, jd=None, filetype='uvh5',
                        dir_path=None, file_re=None):
    """
    Make a glob of visibility files given an observing season and JD.


    This function takes an observing season and a Julian Date, or a 
    path to a directory containing visibility files, and makes a sorted 
    glob of all of the visibility files pertaining to the specified 
    parameters. Specifying ``dir_path`` and ``file_re`` offers the 
    greatest control over which files are loaded in and may completely 
    specify which files to extract; not specifying these parameters 
    results in the function making these decisions, based on the 
    structure of the file storage on lustre at the time of writing.
    Note that specifying ``dir_path`` and ``file_re`` *should* allow 
    you to use this function on a local machine, or on a different 
    filepath on whichever computing cluster you are using.

    Parameters
    ----------
    obs_season : str
        Observing season; currently supported options are 'H1C', 'H2C', 
        and 'H3C'. If specifying 'H1C', then you must append the appropriate 
        IDR tag ('_IDR1', '_IDR2', '_IDR3') to ``obs_season`` before 
        passing the parameter to this function. This parameter does not 
        need to be specified if ``dir_path`` is specified.

    jd : int or str
        Julian Date for which to extract observational data. This must be 
        specified if ``obs_season`` is also specified. If ``dir_path`` and 
        ``file_re`` are specified, then this is ignored.

    filetype : str, optional
        Extension used for the visibility files; must be a type supported 
        by ``pyuvdata``. Default is 'uvh5'.

    dir_path : str, optional
        Path to the directory containing the visibility files, may be 
        absolute or relative (use relative paths at your own risk). Must 
        be specified if ``obs_season`` and ``jd`` are not specified.

    file_re : str or re.Pattern
        String to construct a regular expression for identifying files, 
        or a compiled re.Pattern object. Ideally, this is specified if 
        ``dir_path`` is specified; however, ``dir_path`` and ``jd`` 
        together (perhaps with ``filetype``) should suffice if the file 
        naming convention follows 'zen.JD.HH.uvh5'.

    Returns
    -------
    vis_file_glob : list of str
        Sorted list of visibility files.

    Raises
    ------
    FileNotFoundError
        Raised if ``vis_file_glob`` is empty.
    """
    params = [obs_season, jd, dir_path]
    if all([param is None for param in params]):
        warnings.warn(
            "None of the necessary parameters for retrieving files "
            "were specified. Returning an empty list."
        )
        return []

    # XXX rewrite this at a later date to use the Librarian?
    insufficient_info_err_msg = \
        "Not enough information to construct the file glob. Please " \
        "refer to the documentation for this function to determine " \
        "which parameters must be specified."
    if dir_path is None:
        if obs_season is None or jd is None:
            raise ValueError(insufficient_info_err_msg)

        dir_path = "/lustre/aoc/projects/hera/{obs_season}/{jd}"
        if obs_season == "H1C_IDR3":
            obs_season = "H1C_IDR3/IDR3_1"
        dir_path.format(obs_season=obs_season, jd=jd)
        
        # sometimes things are nested... see some H3C directories
        if glob.glob(os.path.join(dir_path, "zen.*")) == []:
            dir_path = os.path.join(dir_path, "{jd}").format(jd=jd)

    if file_re is None:
        # make sure we have enough information to find the files
        if jd is None:
            raise ValueError(insufficient_info_err_msg)
        file_prefix = "zen.{jd}".format(jd=jd)
        # update this if we go back to including HH in file names
        if int(jd) >= 2458838:
            file_re = re.compile(
                file_prefix + ".[0-9]{5}." + filetype
            )
        else:
            file_re = re.compile(
                file_prefix + ".[0-9]{5}.HH." + filetype
            )
    elif isinstance(file_re, str):
        file_re = re.compile(file_re)
    elif isinstance(file_re, re.Pattern):
        pass
    else:
        raise TypeError("Unsupported type for ``file_re``.")

    # this is kind of lazy, but whatever
    vis_file_glob = glob.glob(dir_path)
    vis_file_glob = sorted(
        [
            vis_file for vis_file in vis_file_glob 
            if file_re.findall(vis_file) != []
        ]
    )

    if vis_file_glob == []:
        raise FileNotFoundError(
            "No visibility files were found. Please check your settings."
        )

    return vis_file_glob


def uvd_downselect(uvd, use_ants=None, use_bls=None, use_pols='linear',
                   use_autos=True, use_cross=False, use_times=None,
                   use_freqs=None):
    """Select only a subset of the data in ``uvd``.

    XXX refer to numpydoc style (does UVData etc receive ``...``?)
    Parameters
    ----------
    uvd : UVData or list of UVData
        UVData object, or path to a file that may be read by a UVData object, 
        on which to perform the data reduction. A list of UVData objects or 
        strings may also be passed, but the list must be of uniform type. If 
        a list is passed, then *all* UVData objects are loaded in a single 
        UVData object.

    use_ants : array-like of int, optional
        List of antenna numbers whose data should be kept. Default is to 
        keep data for all antennas.

    use_bls : array-like of 2- or 3-tuples, optional
        List of antenna pairs or baseline tuples specifying which baselines 
        (and possibly polarizations) to keep. Default is to keep all baselines.

    use_pols : str or array-like, optional
        If passing a string, then it must be one of the following: 'linear', 
        'cross', or 'all' (these refer to which visibility polarizations to 
        keep). If passing an array-like object, then the entries must either 
        be polarization strings or polarization integers. Polarization strings 
        are automatically converted to polarization integers. Default is to 
        use only the linear polarizations ('xx' and 'yy' or 'ee' and 'nn').

    use_autos : bool, optional
        Whether to keep the autocorrelations. Default is to keep the autos.

    use_cross : bool, optional
        Whether to keep the cross-correlations. Default is to discard the 
        cross-correlations.

    use_times : array-like, optional
        Times to keep. If length-2, then it is interpreted as a range of 
        time values and must be specified in Julian Date. Otherwise, the 
        times must exist in the ``time_array`` attribute of the ``UVData`` 
        object corresponding to ``uvd``. Default is to use all times.

    use_freqs : array-like, optional
        Frequencies or frequency channels to keep. If each entry is an 
        integer, then it is interpreted as frequency channels. Default 
        is to use all frequencies.

    Returns
    -------
    uvd : UVData
        UVData object downselected according to the parameters chosen.
    """
    # handle different types for ``uvd``
    if isinstance(uvd, str):
        uvd_ = uvd
        uvd = UVData()
        uvd = uvd.read(uvd_)
    elif isinstance(uvd, UVData):
        pass
    elif isinstance(uvd, (list, tuple)):
        if all([isinstance(uvd_, str) for uvd_ in uvd]):
            uvd_ = uvd
            uvd = UVData()
            uvd = uvd.read(uvd_)
        elif all([isinstance(uvd_, UVData) for uvd_ in uvd]):
            _uvd = uvd[0]
            for uvd_ in uvd[1:]:
                _uvd += uvd_
            uvd = _uvd
        else:
            raise ValueError(
                "If you pass a list or tuple for ``uvd``, then every entry "
                "in the list must be of the same type (str or UVData)."
            )
    else:
        raise ValueError(
            "``uvd`` must be either a string, UVData object, or list/tuple "
            "of strings or UVData objects (no mixing of types allowed)."
        )

    # first downselect: polarization
    # XXX can make a helper function for this
    pol_strings = uvd.get_pols()
    pol_array = uvd.polarization_array
    if use_pols == 'linear':
        use_pols = [
            polstr2num(pol) for pol in ('xx', 'yy')
#            polstr2num(pol) for pol in pol_strings
#            if pol[0] == pol[1]
        ]
    elif use_pols == 'cross':
        use_pols = [
            polstr2num(pol) for pol in ('xy', 'yx')
#            polstr2num(pol) for pol in pol_strings
#            if pol[0] != pol[1]
        ]
    elif use_pols == 'all':
        use_pols = pol_array
    else:
        try:
            _ = iter(use_pols)
        except TypeError:
            raise ValueError(
                "``use_pols`` must be one of the following:\n"
                "'linear' : use only linear polarizations\n"
                "'cross' : use only cross polarizations\n"
                "'all' : use all polarizations\n"
                "iterable of polarization numbers"
            )
        try:
            use_pols = [
                polstr2num(pol) if isinstance(pol, str)
                else pol for pol in use_pols
            ]
            if not all([type(pol) is int for pol in use_pols]):
                raise KeyError
        # in case polarizations aren't recognized
        except KeyError:
            warnings.warn(
                "Polarizations not recognized. Skipping polarization downselect."
            )
            use_pols = pol_array
    
    # actually downselect along polarization
    if use_pols != pol_array:
        uvd.select(polarizations=use_pols, keep_all_metadata=False)

    # next downselect: visibility type
    if use_autos and not use_cross:
        ant_str = 'auto'
    elif not use_autos and use_cross:
        ant_str = 'cross'
    else:
        ant_str = 'all'
    if ant_str != 'all':
        uvd.select(ant_str=ant_str, keep_all_metadata=False)

    # next downselect: antennas
    if use_ants is not None:
        uvd.select(antenna_nums=use_ants, keep_all_metadata=False)

    # next downselect: baselines
    if use_bls is not None:
        uvd.select(bls=use_bls, keep_all_metadata=False)

    # next downselect: frequency
    if use_freqs is not None:
        if all([isinstance(freq, int) for freq in use_freqs]):
            uvd.select(freq_chans=use_freqs, keep_all_metadata=False)
        else:
            uvd.select(frequencies=use_freqs, keep_all_metadata=False)

    # next downselect: time
    if use_times is not None:
        if len(use_times) == 2:
            uvd.select(time_range=use_times, keep_all_metadata=False)
        else:
            uvd.select(times=use_times, keep_all_metadata=False)

    # all the downselecting should be done at this point
    return uvd

