import numpy as np
from pyuvdata import UVData
from hera_qm import xrfi
import matplotlib.pyplot as plt
import glob as glob
import re
import aipy
import numpy.ma as ma
SPEED_OF_LIGHT = 299792458.
import numpy.fft as fft
import scipy.signal as signal
from pyuvdata import utils as pyuvutils
import copy
KBOLTZMANN = 1.38e-23
JY = 1e-26
SPEED_OF_LIGHT = 3e8
import scipy.integrate as integrate
import scipy.special as sp
import scipy.interpolate as interp
import aipy.deconv as deconv
memodir = './'
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from uvtools import dspec
from scipy.signal import windows
from warnings import warn
from scipy.optimize import leastsq, lsq_linear
import multiprocessing
from astropy.time import Time, TimezoneInfo
import astropy.units as units
NCPU = multiprocessing.cpu_count()
from aipy.deconv import clean
try:
    from joblib import Parallel, delayed
    PARALLELIZED = True
except:
    PARALLELIZED = False
    print('Parallelization not supported. Install joblib to enable parallelization.')
PARALLELIZED = False #Turn on false for now. This isn't working for aipy clean :(
'''
The following methods are modified versions of the ones appearing in uvtools.
'''

def calc_width(filter_size, real_delta, nsamples):
    '''Calculate the upper and lower bin indices of a fourier filter
    Arguments:
        filter_size: the half-width (i.e. the width of the positive part) of the region in fourier
            space, symmetric about 0, that is filtered out. In units of 1/[real_delta].
            Alternatively, can be fed as len-2 tuple specifying the absolute value of the negative
            and positive bound of the filter in fourier space respectively.
            Example: (20, 40) --> (-20 < tau < 40)
        real_delta: the bin width in real space
        nsamples: the number of samples in the array to be filtered
    Returns:
        uthresh, lthresh: bin indices for filtered bins started at uthresh (which is filtered)
            and ending at lthresh (which is a negative integer and also not filtered).
            Designed for area = np.ones(nsamples, dtype=np.int); area[uthresh:lthresh] = 0
    '''
    if isinstance(filter_size, (list, tuple, np.ndarray)):
        _, l = calc_width(np.abs(filter_size[0]), real_delta, nsamples)
        u, _ = calc_width(np.abs(filter_size[1]), real_delta, nsamples)
        return (u, l)
    bin_width = 1.0 / (real_delta * nsamples)
    w = int(np.around(filter_size / bin_width))
    uthresh, lthresh = w + 1, -w
    if lthresh == 0:
        lthresh = nsamples
    return (uthresh, lthresh)

def gen_window(window, N, alpha=0.5, edgecut_low=0, edgecut_hi=0, **kwargs):
    """
    Generate a 1D window function of length N.
    Args:
        window : str, window function
        N : int, number of channels for windowing function.
        edgecut_low : int, number of bins to consider as zero-padded at the low-side
            of the array, such that the window smoothly connects to zero.
        edgecut_hi : int, number of bins to consider as zero-padded at the high-side
            of the array, such that the window smoothly connects to zero.
        alpha : if window is 'tukey', this is its alpha parameter.
    """
    # parse multiple input window or special windows
    w = np.zeros(N, dtype=np.float)
    Ncut = edgecut_low + edgecut_hi
    if Ncut >= N:
        raise ValueError("Ncut >= N for edgecut_low {} and edgecut_hi {}".format(edgecut_low, edgecut_hi))
    if edgecut_hi > 0:
        edgecut_hi = -edgecut_hi
    else:
        edgecut_hi = None
    if window in ['none', None, 'None', 'boxcar', 'tophat']:
        w[edgecut_low:edgecut_hi] = windows.boxcar(N - Ncut)
    elif window in ['blackmanharris', 'blackman-harris']:
        w[edgecut_low:edgecut_hi] =  windows.blackmanharris(N - Ncut)
    elif window in ['hanning', 'hann']:
        w[edgecut_low:edgecut_hi] =  windows.hann(N - Ncut)
    elif window == 'tukey':
        w[edgecut_low:edgecut_hi] =  windows.tukey(N - Ncut, alpha)
    else:
        try:
            # return any single-arg window from windows
            w[edgecut_low:edgecut_hi] = getattr(windows, window)(N - Ncut)
        except AttributeError:
            raise ValueError("Didn't recognize window {}".format(window))
    #w = w / np.sqrt(np.mean(w**2.))
    w = w / w.mean()
    return w

def sqrt_abs(x, negatives = False):
    '''
    sqrut of absolute value of x
    '''
    output = np.sqrt(np.abs(x))
    if negatives:
        negative_vals = x <= 0.
        output[negative_vals] = -output[negative_vals]
    return output

def generate_sum_diff(uvd):
    '''
    Generate the sum and difference data from a single pyuvdat object.
    Args:
        uvd, pyuvdata object.
    Returns:
        uvd_sum, pyuvdata object: uvd_sum is the sum of even and odd time steps.
        uvd_diff, pyuvdata object: uvd_diff is the diff of even and odd time steps.
    '''
    times = np.unique(uvd.time_array)
    if np.mod(len(times),2) == 1:
        uvd = uvd.select(times = times[:-1],inplace=False)
        times = times[:-1]

    even_times = times[::2]
    odd_times = times[1::2]
    uvd_even = uvd.select(times = even_times,inplace=False)
    uvd_odd = uvd.select(times = odd_times,inplace=False)
    uvd_diff = copy.deepcopy(uvd_even)
    uvd_sum = copy.deepcopy(uvd_even)
    uvd_diff.data_array = uvd_even.data_array - uvd_odd.data_array
    uvd_sum.data_array = uvd_odd .data_array + uvd_even.data_array
    return uvd_sum, uvd_diff


def clean_filter(data, wgts, clean_area_centers,clean_area_widths, real_delta, fringe_rate_width = None, clean2d=False, tol=1e-9, window='none',
                             skip_wgt=0.1, maxiter=100, gain=0.1, filt2d_mode='rect', alpha=0.5,
                             edgecut_low=0, edgecut_hi=0, add_clean_residual=False,bad_resid = False, bad_wghts = False):
    '''Apply a highpass fourier filter to data. Uses aipy.deconv.clean. Default is a 1D clean
    on the last axis of data.
    Arguments:
        data: 1D or 2D (real or complex) numpy array to be filtered.
            (Unlike previous versions, it is NOT assumed that weights have already been multiplied
            into the data.)
        wgts: real numpy array of linear multiplicative weights with the same shape as the data.
        area_centers: a list of centers for clean windows in units of 1/(real_delta units)
        area_widths: a list of widths of clean windows in units of 1/(real_delta units)
        real_delta: the bin width in real space of the dimension to be filtered.
        fringe_rate_width: the width of the fringe rate filter in units of 1/(real delta units)
            If 2D cleaning, then real_delta must also be a len-2 list.
        clean2d : bool, if True perform 2D clean, else perform a 1D clean on last axis.
        tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
        window: window function for filtering applied to the filtered axis.
            See dspec.gen_window for options. If clean2D, can be fed as a list
            specifying the window for each axis in data.
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Only works properly when all weights are all between 0 and 1.
        maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
        gain: The fraction of a residual used in each iteration. If this is too low, clean takes
            unnecessarily long. If it is too high, clean does a poor job of deconvolving.
        alpha : float, if window is 'tukey', this is its alpha parameter.
        filt2d_mode : str, only applies if clean2d == True. options = ['rect', 'plus']
            If 'rect', a 2D rectangular filter is constructed in fourier space (default).
            If 'plus', the 'rect' filter is first constructed, but only the plus-shaped
            slice along 0 delay and fringe-rate is kept.
        edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
            such that the windowing function smoothly approaches zero. For 2D cleaning, can
            be fed as a tuple specifying edgecut_low for first and second FFT axis.
        edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
            such that the windowing function smoothly approaches zero. For 2D cleaning, can
            be fed as a tuple specifying edgecut_hi for first and second FFT axis.
        add_clean_residual : bool, if True, adds the CLEAN residual within the CLEAN bounds
            in fourier space to the CLEAN model. Note that the residual actually returned is
            not the CLEAN residual, but the residual in input data space.
    Returns:
        d_mdl: CLEAN model -- best fit low-pass filter components (CLEAN model) in real space
        d_res: CLEAN residual -- difference of data and d_mdl, nulled at flagged channels
        info: dictionary (1D case) or list of dictionaries (2D case) with CLEAN metadata
    '''
    # type checks
    dndim = data.ndim
    assert dndim == 1 or dndim == 2, "data must be a 1D or 2D ndarray"
    if clean2d:
        assert dndim == 2, "data must be 2D for 2D clean"
        assert isinstance(filter_size, (tuple, list)), "filter_size must be list or tuple for 2D clean"
        assert len(filter_size) == 2, "len(filter_size) must equal 2 for 2D clean"
        assert isinstance(filter_size[0], (int, np.integer, float, np.float, list, tuple)) \
            and isinstance(filter_size[1], (int, np.integer, float, np.float, list, tuple)), "elements of filter_size must be floats or lists"
        assert isinstance(real_delta, (tuple, list)), "real_delta must be list or tuple for 2D clean"
        assert len(real_delta) == 2, "len(real_delta) must equal 2 for 2D clean"
        if isinstance(edgecut_low, (int, np.integer)):
            edgecut_low = (edgecut_low, edgecut_low)
        if isinstance(edgecut_hi, (int, np.integer)):
            edgecut_hi = (edgecut_hi, edgecut_hi)
        if isinstance(window, (str, np.str)):
            window = (window, window)
        if isinstance(alpha, (float, np.float, int, np.integer)):
            alpha = (alpha, alpha)
    else:
        assert isinstance(real_delta, (int, np.integer, float, np.float)), "if not clean2d, real_delta must be a float"
        assert isinstance(window, (str, np.str)), "If not clean2d, window must be a string"

    # 1D clean
    if not clean2d:
        # setup _d and _w arrays
        win = gen_window(window, data.shape[-1], alpha=alpha, edgecut_low=edgecut_low, edgecut_hi=edgecut_hi)
        if dndim == 2:
            win = win[None, :]
        _d = np.fft.ifft(data * wgts * win, axis=-1)
        if bad_wghts:
            _w = np.fft.ifft(wgts * win, axis=-1)
        else:
            _w = np.fft.ifft(wgts, axis=-1)

        # calculate area array
        delays = fft.fftfreq(data.shape[-1],real_delta)
        #print(delays)
        area = np.zeros(data.shape[-1],dtype=np.int)
        for dc,dw in zip(clean_area_centers, clean_area_widths):
            #print(dc,dw)
            area[np.abs(delays-dc) <= dw] = 1

        #print('clean area?')
        #print(area)
        # run clean
        if dndim == 1:
            # For 1D data array run once
            _d_cl, info = aipy.deconv.clean(_d, _w, area=area, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
            _d_res = info['res']
            del info['res']

        elif data.ndim == 2:
            # For 2D data array, iterate
            info = []
            _d_cl = np.empty_like(data)
            _d_res = np.empty_like(data)
            if not PARALLELIZED:
                for i in range(data.shape[0]):
                    if _w[i, 0] < skip_wgt:
                        _d_cl[i] = 0  # skip highly flagged (slow) integrations
                        _d_res[i] = _d[i]
                        info.append({'skipped': True})
                    else:
                        _cl, info_here = clean(_d[i], _w[i], area=area, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
                        _d_cl[i] = _cl
                        _d_res[i] = info_here['res']
                        del info_here['res']
                        info.append(info_here)
            else:
                print('Parallelized!')
                nfreq = data.shape[0]
                parallel_out = Parallel(n_jobs=NCPU)(delayed(clean)(_d[i], _w[i], area=area, tol=tol,
                 stop_if_div=False, maxiter=maxiter, gain=gain) for i in range(data.shape[0]))
                for i in range(data.shape[0]):
                     _cl, info_here = parallel_out[i]
                     _d_cl[i] = _cl
                     _d_res[i] = info_here['res']
                     del info_here['res']
                     info.append(info_here)

    # 2D clean on 2D data
    else:
        # setup _d and _w arrays
        win1 = gen_window(window[0], data.shape[0], alpha=alpha[0], edgecut_low=edgecut_low[0], edgecut_hi=edgecut_hi[0])
        win2 = gen_window(window[1], data.shape[1], alpha=alpha[1], edgecut_low=edgecut_low[1], edgecut_hi=edgecut_hi[1])
        win = win1[:, None] * win2[None, :]
        _d = np.fft.ifft2(data * wgts * win, axes=(0, 1))
        if bad_wghts:
            _w = np.fft.ifft2(wgts * win, axes=(0, 1))
        else:
            _w = np.fft.ifft2(wgts, axes=(0, 1))

        # calculate area array

        delays = fft.fftfreq(data.shape[-1],real_delta[-1])
        a2 = np.zeros(data.shape[-1],dtype=np.int)
        for dc,dw in zip(clean_area_centers, clean_area_widths):
            a2[np.abs(delays-dc) <= dw] = 1
        times = fft.fftfreq(data.shape[0],real_delta[0])
        a1 = np.zeros(data.shape[-2],dtype=np.int)
        a1[np.abs(times)<=fringe_rate_width] = 1
        area = np.outer(a1, a2)

        # check for filt2d_mode
        if filt2d_mode == 'plus':
            _area = np.zeros(data.shape, dtype=np.int)
            _area[:, 0] = area[:, 0]
            _area[0, :] = area[0, :]
            area = _area
        elif filt2d_mode == 'rect':
            pass
        else:
            raise ValueError("Didn't recognize filt2d_mode {}".format(filt2d_mode))

        # run clean
        _d_cl, info = aipy.deconv.clean(_d, _w, area=area, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
        _d_res = info['res']
        del info['res']

    # add resid to model in CLEAN bounds
    if add_clean_residual:
        _d_cl += _d_res * area

    # fft back to input space
    if clean2d:
        d_mdl = np.fft.fft2(_d_cl, axes=(0, 1))
        d_res = np.fft.fft2(_d_res, axes=(0, 1))
    else:
        d_mdl = np.fft.fft(_d_cl)
        d_res = np.fft.fft(_d_res)

    # get residual in data space
    if bad_resid:
        d_res = (data - d_mdl) * ~np.isclose(wgts * win, 0.0)

    return d_mdl, d_res, info


def down_select_data(data,fmin=45e6,fmax=85e6,lst_min=None,lst_max=None):
    '''
    data, pyuvdata object storing summed measurement
    fmin, minimum frequency (Hz), float
    fmax, maximum frequency (Hz), float
    lst_min, minimum lst to run waterfall from -- !!BREAKS IF DATA CROSSES 0 LST!!
    lst_max, maximum lst to run waterfall from -- !!BREAKS IF DATA CROSSES 0 LST!!
    '''

    if lst_min is None:
        lst_min = np.unique(data.lst_array).min() * 24. / 2. /np.pi
    if lst_max is None:
        lst_max = np.unique(data.lst_array).max() * 24. / 2. / np.pi
    '''
    This LST selection will break if the data crosses midnight.
    '''
    #print('max lst provided is %f'%(lst_max))
    #print('min lst provided is %f'%(lst_min))
    lst_inds = np.unique(data.lst_array, return_index=True)[1]
    #print(lst_inds.shape)
    #print(lst_inds)
    lst_unique = np.asarray([data.lst_array[index] for index in sorted(lst_inds)])
    
    lst_select = np.logical_and(lst_unique >=  lst_min * 2. * np.pi / 24.,
                                lst_unique <= lst_max * 2. * np.pi / 24.)
    times_select = np.unique(data.time_array)[lst_select]
    #print('max lst select is %f'%(lst_unique[lst_select].max()*12/np.pi))
    #print('min lst select is %f'%(lst_unique[lst_select].min()*12/np.pi))
    ntimes = len(times_select)
    #print('ntimes init = %d'%ntimes)
    if np.mod(ntimes,2)==1 and ntimes>1:
        ntimes -=1
        times_select = times_select[:-1]

    #Select frequencies between fmin and fmax
    freqs = data.freq_array.squeeze()
    select_channels = np.where(np.logical_and(freqs>=fmin,freqs<=fmax))[0]
    if np.mod(len(freqs[select_channels]),2)==1:
        select_channels=select_channels[:-1]
    data = data.select(freq_chans=select_channels,times = times_select,inplace=False)
    #print('ntimes after = %d'%ntimes)
    #print('max time resulting = %f'%(data.lst_array.max()*12/np.pi))
    return data

def get_corr_data(data,corrkey, f_threshold = None, t_threshold = None,return_xy = False,
percentile_flags = False, flagp_min=2, flagp_max = 98, fmin=None,fmax=None,lstmin=None,lstmax=None):
    '''
    retrieve data and flags from uv data for a single polarization
    and antenna pair.
    Args:
        data, pyuvdata object
        corrkey, 3-tuple (ant0, ant1, pol)
        t_threshold, optional float , if fraction of flagged channels at single time is a above this, flag entire time.
        f_threshold, optional float, if fraction of flagged channels at a single freq is above this, flag entire freq at all times.
        f_int, 2-list of frequency bounds to select
        lst_int, 2-list of lst intervals to select
    Returns:
        ntimes x nfreq array of bools, ntimes x nfreq array of complex128
        first array is flags, second array is visibility.
    '''
    ant1,ant2,pol = corrkey[0], corrkey[1], corrkey[2]
    #select baselines
    selection = data._key2inds((ant1, ant2))
    if len(selection[0])==0:
        selection = selection[1]
        a1 = ant2
        a2 = ant1
    elif len(selection[1])==0:
        selection = selection[0]
        a1 = ant1
        a2 = ant2
    else:
        raise ValueError("Correlation between antennas %d x %d not present in data.")

    f_int = [fmin,fmax]
    lst_int = [lstmin,lstmax]
    darray = data.data_array[selection,:,:,pol].squeeze()
    wghts = data.flag_array[selection,:,:,pol].squeeze()

    if f_int[0] is None or f_int[0] <= data.freq_array.min() or f_int[0] >= data.freq_array.max():
        f_int[0] = data.freq_array.min()
    if f_int[1] is None or f_int[1] >= data.freq_array.max() or f_int[1] <= data.freq_array.min():
        f_int[1] = data.freq_array.max()
    if lst_int[0] is None or lst_int[0] <= data.lst_array.min() or lst_int[0] >= data.lst_array.max():
        lst_int[0] = data.lst_array.min()
    if lst_int[1] is None or lst_int[1] >= data.lst_array.max() or lst_int[1] <= data.lst_array.min():
        lst_int[1] = data.lst_array.max()

    #print('after selection shape')
    #print(darray.shape)
    lsts = np.unique(data.lst_array)
    fqs = data.freq_array.squeeze()

    lst_select = np.logical_and(lsts>=lst_int[0], lsts<= lst_int[1])
    fqs_select = np.logical_and(fqs>=f_int[0], fqs<= f_int[1])
    lsts = lsts[lst_select]
    fqs = fqs[fqs_select]
    #print(len(lsts))
    #print(len(fqs))
    wghts, darray = wghts[:,fqs_select][lst_select,:], darray[:,fqs_select][lst_select,:]

    #print('after further selection')
    #print(darray.shape)

    #print(len(lsts))
    if not t_threshold is None:
        for tnum in range(len(lsts)):
            if float(len(wghts[tnum,wghts[tnum, :]]))/len(wghts[tnum, :]) >= t_threshold:
                wghts[tnum,:] = True #flag entire time

    if not f_threshold is None:
        for cnum in range(len(fqs)):
            if float(len(wghts[wghts[:, cnum],cnum]))/len(wghts[:, cnum]) >= f_threshold:
                wghts[:, cnum] = True #flag entire channel

#perform percentile flagging.
    flag_percentiles = [flagp_min, flagp_max]
    if percentile_flags:
        percentiles = np.percentile(np.abs(darray)[np.invert(wghts)],flag_percentiles)
        wghts[np.logical_or(np.abs(darray)<=percentiles[0],np.abs(darray)>=percentiles[1])] = True


    del f_int
    del f_threshold
    del lst_int
    del t_threshold

    if not return_xy:
        return wghts,darray
    else:
        return lsts, fqs,wghts, darray

def get_horizon(data, corrkey):
    '''
    get horizon max for baseline (ant0, ant1) given by corrkey
    Args:
        data, pyuvdata object
        corrkey, 2-tuple with two antenna numbers
    Returns:
        float, maximum delay in data set for baseline given by corrkey
    '''
    ant1,ant2 = corrkey[0], corrkey[1]
    #select baselines
    selection = data._key2inds((ant1, ant2))
    if len(selection[0])==0:
        selection = selection[1]
    elif len(selection[1])==0:
        selection = selection[0]
    else:
        raise ValueError("(%d, %d) is not present in the data set"%(corrkey[0],corrkey[1]))

    horizon = np.linalg.norm(data.uvw_array[selection,:],axis=1).max()
    return horizon / SPEED_OF_LIGHT



def filter_data_clean(corrkey,data,data_d = None,fmin = 45e6, fmax = 85e6, manual_flags = [], manual_override_flags = False,
                     fringe_rate_max = .2e-3, area_centers = [0.], area_widths = [600e-9], normalize_average = False,
                     lst_min = None,lst_max=None,taper='boxcar',filt2d_mode='rect', lst_norm_min = None, lst_norm_max = None,
                     add_clean_components=True,freq_domain = 'delay', extra_chan_flags = [],
                     time_domain = 'time',tol=1e-7,bad_wghts=False, f_threshold = 0.1,
                     flag_across_time = True,bad_resid=False, t_threshold = .1, norm_zero_delay = False,
                     fringe_rate_filter = False,acr=False, positive_delay_only = False):
    '''
    data, pyuvdata object storing summed measurement
    data_d, pyuvdata object storing differential measurement
    corrkey, tuple for correlation (ant1,ant2,pol)
    fmin, minimum frequency (Hz), float
    fmax, maximum frequency (Hz), float
    fringe_rate_max, maximum fringe_rate to clean (Hz), float, default = .2e-3 sec
    area_widths, a list of half widths for cleaning windows.
    area_centers, a list of delays around which cleaning will center.
    lst_min, minimum lst to run waterfall from -- !!BREAKS IF DATA CROSSES 0 LST!!
    lst_max, maximum lst to run waterfall from -- !!BREAKS IF DATA CROSSES 0 LST!!
    taper, string, Fourier window function.
    normalize_average, if True, divided data and diff data by average
    add_clean_components, bool, if True, returnb resid + clean components
                                if False, return only resid.
    freq_domain, specify if the output should be in the "frequency" or "delay" domain
                string
    time_domain, specify if the output should be in the "time" or "fringe-rate" domain
                string
    flag_across_time, if True, flags in frequency are the union of all flags
                      across time at that frequency
    fringe_rate_filter, if True, clean in 2d and filter out fringe-rate modes.
                        if False, only clean per time in delay space.
    t_threshold, if fraction of flagged channels at single time is a above this,
                flag entire time.
    f_threshold, if fraction of flagged channels at a single freq is above this,
                flag entire freq at all times.
    manual_flags, list of 2-tuples with centers and widths of flagging windows.
    manual_override_flags, flags in manual_flags override pre-existing flags.
    extra_chan_flags, list of ints containing channel numbers to entirely flag.
    norm_zero_delay, normalize by zero delay component independently for each time.
    '''
    if data_d is None:
        data, data_d = generate_sum_diff(data)

    if lst_norm_min is None:
        lst_norm_min = lst_min
    if lst_norm_max is None:
        lst_norm_max = lst_max

    data_norm = down_select_data(data,fmin,fmax,lst_norm_min,lst_norm_max)
    data = down_select_data(data,fmin,fmax,lst_min,lst_max)
    data_d = down_select_data(data_d,fmin,fmax,lst_min,lst_max)
    #print('data_d shape = %s'%str(data_d.data_array.shape))
    #print('data_d shape')
    #print('data shape')
    #print(data_d.data_array.shape)
    #print(data.data_array.shape)
    if tol == 0:
        raise ValueError("Invalid Tolerance of 0. Provided!")


    freqs = data.freq_array.squeeze()
    delays = fft.fftshift(fft.fftfreq(len(freqs),freqs[1]-freqs[0]))
    lsts = np.unique(data.lst_array)*12/np.pi
    times = 3600.*24.*np.unique(data.time_array)
    fringe_rates = fft.fftshift(fft.fftfreq(len(times),times[1]-times[0]))
    times_norm = 3600.*24.*np.unique(data_norm.time_array)
    #print('ntimes = %d'%len(times))
    #print('from nblts and nbls = %d'%int(data.data_array.shape[0]/data.Nbls))

    wghts,darray = get_corr_data(data, corrkey)
    _, darray_d = get_corr_data(data_d, corrkey)
    wghts_norm, darray_norm = get_corr_data(data_norm,corrkey)
    #print(darray.shape[0])


    if manual_override_flags:
        wghts[:] = False
        wghts_norm[:] = False
    wghts_c = copy.copy(wghts)
    wghts_nc = copy.copy(wghts_norm)

    if flag_across_time:
        for fnum in range(len(freqs)):
            wghts[:,fnum] = np.any(wghts[:,fnum])
            wghts_norm[:,fnum] = np.any(wghts_norm[:,fnum])
    #print(times.shape)
    #print(wghts.shape)
    #print(wghts.shape)
    for tnum in range(len(times)):
        if float(len(wghts_c[tnum,wghts_c[tnum, :]]))/len(wghts_c[tnum, :]) >= t_threshold:
            wghts[tnum,:] = True #flag entire time

    for tnum in range(len(times_norm)):
        if float(len(wghts_nc[tnum,wghts_nc[tnum, :]]))/len(wghts_nc[tnum, :]) >= t_threshold:
            wghts_norm[tnum,:] = True #flag entire time


    for cnum in range(len(freqs)):
        if float(len(wghts_c[wghts[:, cnum],cnum]))/len(wghts_c[:, cnum]) >= f_threshold:
            wghts[:, cnum] = True #flag entire channel
    for cnum in range(len(freqs)):
        if float(len(wghts_nc[wghts_nc[:, cnum],cnum]))/len(wghts_nc[:, cnum]) >= f_threshold:
            wghts_norm[:, cnum] = True #flag entire channel


    for cnum in extra_chan_flags:
        wghts[:, cnum] = True
        wghts_norm[:,cnum] = True

    for fc,fw in manual_flags:
        flag_temp = np.abs(freqs-fc) <= fw
        wghts[:,flag_temp] = True
        wghts_norm[:,flag_temp] = True

    if normalize_average:
        #factor of 2 because data array is even + odd
        norm_factor = 2. / np.mean(np.abs(darray_norm[np.invert(wghts_norm)]))
        darray = darray * norm_factor
        darray_d = darray_d * norm_factor


    if norm_zero_delay:
        norm_taper = windows.get_window(taper, len(freqs))
        norm_taper /= norm_taper.mean()
        for tnum in range(len(times)):
            x = norm_taper * darray[tnum]
            x[wghts[tnum]] = 0.
            norm_factor = np.abs(fft.ifft(x))[0]
            darray[tnum] = darray[tnum] /norm_factor
            darray_d[tnum] = darray_d[tnum] /norm_factor

    wghts = np.invert(wghts).astype(float)

    if isinstance(area_widths,float):
        area_widths = [area_widths]
    if isinstance(area_centers,float):
        area_centers = [area_centers]


    if fringe_rate_filter:
        delta_bin = [times[1]-times[0],freqs[1]-freqs[0]]
        model,resid,info = clean_filter(darray,wghts,area_centers,area_widths,delta_bin,fringe_rate_width = max_fringe_rate,
                                                          tol=tol,add_clean_residual=acr,
                                                          clean2d=True,window=taper,filt2d_mode=filt2d_mode,
                                                          bad_resid=bad_resid,bad_wghts=bad_wghts)
        model_d,resid_d,info_d = clean_filter(darray_d,wghts,area_centers, area_widths,  delta_bin, fringe_rate_width = max_fringe_rate,
                                                          tol=tol,add_clean_residual=acr,
                                                          clean2d=True,window=taper,bad_wghts=bad_wghts,
                                                          filt2d_mode=filt2d_mode,bad_resid=bad_resid)
    else:
        delta_bin = freqs[1]-freqs[0]
        model,resid,info = clean_filter(darray,wghts,area_centers, area_widths,
                                                          delta_bin,add_clean_residual=acr,
                                                          bad_wghts = bad_wghts,
                                                          tol=tol,window=taper,bad_resid=bad_resid)

        model_d,resid_d,info_d = clean_filter(darray_d,wghts,area_centers, area_widths,
                                                                delta_bin,add_clean_residual=acr,
                                                                bad_wghts = bad_wghts,
                                                                tol=tol,window=taper,bad_resid=bad_resid)

    #resid = info[0]['res']
    #resid_d = info_d[0]['res']

    if add_clean_components:
        output = model + resid
        output_d = model_d + resid_d
    else:
        output = resid
        output_d = resid_d

    if time_domain == 'fringe-rate':
        y = fringe_rates
        output = fft.fftshift(fft.ifft(fft.fftshift(output,axes=[0]),axis=0),axes=[0])
        output_d = fft.fftshift(fft.ifft(fft.fftshift(output_d,axes=[0]),axis=0),axes=[0])
    else:
        y = np.unique(data.lst_array) * 24. / 2. / np.pi
    if freq_domain == 'delay':
        x = delays
        output = fft.fftshift(fft.ifft(fft.fftshift(output,axes=[1]),axis=1),axes=[1])
        output_d = fft.fftshift(fft.ifft(fft.fftshift(output_d,axes=[1]),axis=1),axes=[1])
        if positive_delay_only:
            output = output[:,int(output.shape[1]/2):]
            output_d = output_d[:,output.shape[1]:]
            x = x[output.shape[1]:]
    else:
        x = freqs
    xg,yg = np.meshgrid(x,y)
    return xg,yg,output,output_d


WMAT_CACHE = {}
def clear_cache():
    WMAT_CACHE = {}

def linear_filter(freqs,ydata,flags,patch_c = [], patch_w = [], filter_factor = 1e-3,weights='I', fourier_taper = None,
                  renormalize=False,zero_flags=True,taper='boxcar',cache = WMAT_CACHE, cmax = 1e16):
    '''
    a linear delay filter that suppresses modes within the wedge by a factor of filter_factor.
    freqs, nchan vector of frequencies
    ydata, nchan vector of complex data
    flags, nchan bool vector of flags
    fourier_taper, taper unfiltered boxcars in fourier domain (to get rid of residual side-lobes).
                   WARNING: This will help get rid of side-lobes of high domain contamination but
                   will mask low-delay structures!
    '''
    if filter_factor == 0:
        weights = 'I'
    if isinstance(patch_c, float):
        patch_c = [patch_c]
    if isinstance(patch_w, float):
        patch_w = [patch_w]
    nf = len(freqs)
    taper=signal.windows.get_window(taper,nf)
    taper/=np.sqrt((taper*taper).mean())

    if weights=='I':
        wmat = np.identity(nf)
        if zero_flags:
            wmat[:,flags]=0.
            wmat[flags,:]=0.
    elif weights == 'WTL':
        wkey = (nf,freqs[1]-freqs[0],filter_factor,zero_flags)+tuple(np.where(flags)[0])\
        + tuple(patch_c) + tuple(patch_w)
        if not wkey in cache:
            fx,fy=np.meshgrid(freqs,freqs)
            cmat_fg = np.zeros_like(fx).astype(complex)
            for pc,pw in zip(patch_c, patch_w):
                cmat_fg += np.sinc(2.*(fx-fy) * pw) * np.exp(2j*np.pi*(fx-fy) * pc)
            cmat = cmat_fg+np.identity(len(freqs))*filter_factor
            #print(cmat.shape)
            #print(flags.shape)
            if zero_flags:
                cmat[:,flags]=0.
                cmat[flags,:]=0.
            wmat = np.linalg.pinv(cmat)*filter_factor
            cache[wkey]=wmat
        else:
            wmat = cache[wkey]

    elif isinstance(weights,np.ndarray) and weights.shape[0] == ydata.shape[0]:
        #allow for manual weights to be provided
        wmat = weights




    output = ydata
    #output = fft.fft(np.dot(wmat,output * taper))
    output = np.dot(wmat,output)
    #output = fft.ifft(output * taper)
    #output = fft.fft(output)/taper
    if not fourier_taper is None:
        #print('Tapering in Fouier Domain!')
        #taper regions outside of filter region.
        nf_filtered = int(2 * patch_w[0] * (freqs[-1] - freqs[0]))

        nf_not_filtered = nf - nf_filtered
        if np.mod(nf_not_filtered,2) == 1:
            nf_filtered += 1
            nf_not_filtered = nf - nf_filtered

        fourier_taper = signal.windows.get_window(fourier_taper,int(nf_not_filtered / 2))
        #fourier_taper = fourier_taper / np.sqrt(np.mean(fourier_taper ** 2. ))
        fourier_taper = fourier_taper / fourier_taper.mean()
        fourier_taper = np.hstack([np.zeros(int(nf_filtered / 2)), fourier_taper]).astype(complex)
        fourier_taper = np.hstack([fourier_taper, fourier_taper])

        output = fft.fft(fourier_taper * fft.ifft(output))

    return output


def filter_data_linear(corrkey,data,data_d = None,fmin = 45e6, fmax = 85e6, norm_zero_delay = False,
                     fringe_rate_max = .2e-3, delay_max = 600e-9, delay_center = 0.,
                     lst_min = None,lst_max=None,taper='boxcar', extra_chan_flags = [],
                     freq_domain = 'delay', zero_flags = True, f_threshold = 0.1, manual_flags = [],
                     manual_override_flags = False, lst_norm_min = None, lst_norm_max = None,
                     time_domain = 'time',tol=1e-7, t_threshold = 0.1, fourier_taper = None,
                     flag_across_time = True, normalize_average = False, positive_delay_only = False,
                       fringe_rate_filter = False, cache = WMAT_CACHE, time_units = 'lst'):
    '''
    data, pyuvdata object storing summed measurement
    data_d, pyuvdata object storing differential measurement, if None, generate diffed data and summed data automatically
    corrkey, tuple for correlation (ant1,ant2,pol)
    fmin, minimum frequency (Hz), float
    fmax, maximum frequency (Hz), float
    fringe_rate_max, maximum fringe_rate to filter (Hz), float, default = .2e-3 sec
    delay_max, maximum delay to clean (sec), float, default = 600e-9 sec
    lst_min, minimum lst to run waterfall from -- !!BREAKS IF DATA CROSSES 0 LST!!
    lst_max, maximum lst to run waterfall from -- !!BREAKS IF DATA CROSSES 0 LST!!
    taper, string, Fourier window function.
    time_units, string, specify 'lst' or 'jd'. 
    freq_domain, specify if the output should be in the "frequency" or "delay" domain
                string
    time_domain, specify if the output should be in the "time" or "fringe-rate" domain
                string
    flag_across_time, if True, flags in frequency are the union of all flags
                      across time at that frequency
    fringe_rate_filter, if True, clean in 2d and filter out fringe-rate modes.
                        if False, only clean per time in delay space.
    normalize_average, if True, normalize data and diff data by average.
    t_threshold, if fraction of flagged channels at single time is a above this, flag entire time.
    f_threshold, if fraction of flagged channels at a single freq is above this, flag entire freq at all times.
    manual_flags, list of 2-tuples with centers and widths of flagging windows.
    manual_override_flags, flags in manual_flags override pre-existing flags.
    extra_chan_flags, list of ints containing channel numbers to entirely flag.
    fourier_taper, taper unfiltered boxcars in fourier domain (to get rid of residual side-lobes).
                   WARNING: This will help get rid of side-lobes of high domain contamination but
                   will mask low-delay structures!
    positive_delay_only, if True, only return positive delays.
    norm_zero_delay, normalize by zero delay component independently for each time.
    Returns:
        xg,yg,output,output_d

    '''
    if data_d is None:
        data, data_d = generate_sum_diff(data)


    if lst_norm_min is None:
        lst_norm_min = lst_min
    if lst_norm_max is None:
        lst_norm_max = lst_max
    if not lst_min is None:
        print('minimum lst = %f'%lst_min)
    if not lst_max is None:
        print('maximum lst = %f'%lst_max)
    data_norm = down_select_data(data, fmin, fmax, lst_norm_min, lst_norm_max)
    data = down_select_data(data,fmin,fmax,lst_min,lst_max)
    data_d = down_select_data(data_d,fmin,fmax,lst_min,lst_max)


    freqs = data.freq_array.squeeze()
    delays = fft.fftshift(fft.fftfreq(len(freqs),freqs[1]-freqs[0]))

    lsts = 12/np.pi * np.unique(data.lst_array)
    times = 3600.*24.*np.unique(data.time_array)
    times_norm = 3600.*24.*np.unique(data_norm.time_array)
    fringe_rates = fft.fftshift(fft.fftfreq(len(times),times[1]-times[0]))
    ntimes = len(times)
    nfreq = len(freqs)

    wghts,darray = get_corr_data(data, corrkey)
    _, darray_d = get_corr_data(data_d, corrkey)
    wghts_norm, darray_norm = get_corr_data(data_norm, corrkey)
    if manual_override_flags:
        wghts[:] = False
        wghts_norm[:]=False

    wghts_c = copy.copy(wghts)
    wghts_nc = copy.copy(wghts_norm)

    if flag_across_time:
        for fnum in range(len(freqs)):
            wghts[:,fnum] = np.any(wghts[:,fnum])
            wghts_norm[:,fnum] = np.any(wghts_norm[:,fnum])
    for tnum in range(len(times)):
        if float(len(wghts_c[tnum,wghts_c[tnum, :]]))/len(wghts_c[tnum, :]) >= t_threshold:
            wghts[tnum,:] = True #flag entire time

    for tnum in range(len(times_norm)):
        if float(len(wghts_nc[tnum,wghts_nc[tnum, :]]))/len(wghts_nc[tnum, :]) >= t_threshold:
            wghts_norm[tnum,:] = True #flag entire time
    for cnum in range(len(freqs)):
        if float(len(wghts_c[wghts[:, cnum],cnum]))/len(wghts_c[:, cnum]) >= f_threshold:
            wghts[:, cnum] = True #flag entire channel
    for cnum in range(len(freqs)):
        if float(len(wghts_nc[wghts_nc[:, cnum],cnum]))/len(wghts_nc[:, cnum]) >= f_threshold:
            wghts_norm[:, cnum] = True #flag entire channel


    for cnum in extra_chan_flags:
        wghts[:, cnum] = True
        wghts_norm[:,cnum] = True
    for fc,fw in manual_flags:
        flag_temp = np.abs(freqs-fc) <= fw
        wghts[:,flag_temp] = True
        wghts_norm[:,flag_temp] = True

    if normalize_average:
        #factor of 2 because data array is even + odd
        norm_factor = 2. / np.mean(np.abs(darray_norm[np.invert(wghts_norm)]))
        darray = darray * norm_factor
        darray_d = darray_d * norm_factor


    if norm_zero_delay:
        norm_taper = windows.get_window(taper, len(freqs))
        norm_taper /= norm_taper.mean()
        for tnum in range(len(times)):
            x = norm_taper * darray[tnum]
            x[wghts[tnum]] = 0.
            norm_factor = np.abs(fft.ifft(x))[0]
            darray[tnum] = darray[tnum] /norm_factor
            darray_d[tnum] = darray_d[tnum] /norm_factor



    if not isinstance(delay_max,list):
        delay_widths = [delay_max]
    if not isinstance(delay_center,list):
        delay_centers = [delay_center]
    resid = np.zeros_like(darray)
    resid_d = np.zeros_like(darray_d)


    if not PARALLELIZED:
        for tnum in range(ntimes):
            if not np.all(wghts[tnum,:]):
                try:
                
                    resid[tnum,:] = linear_filter(freqs,darray[tnum,:],wghts[tnum,:],patch_c = delay_center,
                                                  patch_w = delay_max, filter_factor = tol, weights = 'WTL',
                                                  renormalize = False, zero_flags = zero_flags,
                                                  taper = taper, fourier_taper = fourier_taper)

                    resid_d[tnum,:] = linear_filter(freqs,darray_d[tnum,:],wghts[tnum,:],patch_c = delay_center,
                                                    patch_w = delay_max, filter_factor = tol, weights = 'WTL',
                                                    renormalize = False, zero_flags = zero_flags,
                                                    taper = taper, fourier_taper = fourier_taper)
                except np.linalg.LinAlgError:
                    print('warning: svd not converged.')
                    resid[tnum,:] = np.zeros(darray.shape[1],dtype=complex)
                    wghts[tnum,:] = True
    else:
        print('Parallelized!')
        resid = np.asarray(Parallel(n_jobs = NCPU)(delayed(linear_filter)(freqs,darray[tnum,:],wghts[tnum,:],patch_c = delay_center,
                                     patch_w = delay_max, filter_factor = tol, weights = 'WTL',
                                     renormalize = False, zero_flags = zero_flags,
                                     taper = taper, fourier_taper = fourier_taper) for tnum in range(ntimes)))

        resid_d = np.asarray(Parallel(n_jobs = NCPU)(delayed(linear_filter)(freqs,darray_d[tnum,:],wghts[tnum,:],patch_c = delay_center,
                                     patch_w = delay_max, filter_factor = tol, weights = 'WTL',
                                     renormalize = False, zero_flags = zero_flags,
                                     taper = taper, fourier_taper = fourier_taper) for tnum in range(ntimes)))
    if fringe_rate_filter:
        if not PARALLELIZED:
            for cnum in range(nfreq):
                resid[:,cnum] = linear_filter(times,resid[:,cnum],wghts[:,cnum],patch_c = [0.],
                                         patch_w = [max_fringe_rate], filter_factor = tol, weights = 'WTL',
                                         renormalize = False, zero_flags = zero_flags,
                                         taper = taper, fourier_taper = fourier_taper)
                resid_d[:,cnum] = linear_filter(times,resid_d[:,cnum],wghts[:,cnum],patch_c = [0.],
                                 patch_w = [max_fringe_rate], filter_factor = tol, weights = 'WTL',
                                 renormalize = False, zero_flags = zero_flags,
                                 taper = taper, fourier_taper = fourier_taper)
        else:
            print('Parallelized!')
            resid  = np.asarray(Parallel(n_jobs=NCPU)(delayed(linear_filter)(times,resid[:,cnum],wghts[:,cnum],patch_c = [0.],
                                     patch_w = [max_fringe_rate], filter_factor = tol, weights = 'WTL',
                                     renormalize = False, zero_flags = zero_flags,
                                     taper = taper, fourier_taper = fourier_taper) for cnum in range(nfreq)))
            resid_d  = np.asarray(Parallel(n_jobs=NCPU)(delayed(linear_filter)(times,resid_d[:,cnum],wghts[:,cnum],patch_c = [0.],
                                     patch_w = [max_fringe_rate], filter_factor = tol, weights = 'WTL',
                                     renormalize = False, zero_flags = zero_flags,
                                     taper = taper, fourier_taper = fourier_taper) for cnum in range(nfreq)))

    output = resid
    output_d = resid_d

    if time_domain == 'fringe-rate':
        y = fringe_rates
        taper = signal.windows.get_window(taper, ntimes)
        #taper /= np.sqrt(np.mean(taper**2.))
        taper /= taper.mean()
        taper = np.array([taper for m in range(nfreq)]).T
        output = fft.fftshift(fft.ifft(fft.fftshift(taper * output,axes=[0]),axis=0),axes=[0])
        output_d = fft.fftshift(fft.ifft(fft.fftshift(taper * output_d,axes=[0]),axis=0),axes=[0])
    if time_units == 'lst':
        y = np.unique(data.lst_array) * 24. / 2. / np.pi
    else:
        y = np.unique(data.time_array) 
    if freq_domain == 'delay':
        x = delays
        taper = signal.windows.get_window(taper, nfreq)
        taper = np.array([taper for m in range(ntimes)])
        #taper /= np.sqrt(np.mean(taper**2.))
        taper /= taper.mean()
        output = fft.fftshift(fft.ifft(fft.fftshift(output * taper,axes=[1]),axis=1),axes=[1])
        output_d = fft.fftshift(fft.ifft(fft.fftshift(output_d * taper,axes=[1]),axis=1),axes=[1])
        if positive_delay_only:
            output = output[:,int(output.shape[1]/2):]
            output_d = output_d[:,output.shape[1]:]
            x = x[output.shape[1]:]
    elif freq_domain =='frequency':
        x = freqs
    else:
        raise ValueError("Invalid output domain provided. Must be 'frequency' or 'delay'")
    xg,yg = np.meshgrid(x,y)
    return xg,yg,output,output_d

def integrate_LST(corrkey, data, data_d = None, fmin = 45e6, fmax=85e6, fringe_rate_max = .2e-3, bad_resid = False, bad_wghts = False,
              delay_max = 300e-9, delay_center = 0., lst_min = None, lst_max = None, taper = 'boxcar',
              freq_domain = 'delay', zero_flags = True, normalize_average = False,f_threshold = 0.1, t_threshold = 0.1,
              tol = 1e-7, flag_across_time = True, fringe_rate_filter = False, filter_method = 'linear', fourier_taper = None,
              add_clean_components = True, avg_coherent = True, sq_units = True, cache = WMAT_CACHE):
    '''
    integrate data over LST from a single baseline.
    Args:
        data, pyuvdata object representing data
        data_d, pyuvdata object storing diffed data, if None, generate automatically
        corrkey, key selecting baseline (ant0, ant1, pol)
        fmin, minimum frequency
        fmax, maximum frequency
        normalize_average, if True, than normalize by mean of unflagged data.
        fringe_rate_max, maximum fringe rate to filter below
        filter_method, string set to 'linear' or 'clean' and determines the
                       method to clean at.
        add_clean_components, if True, add clean components to data_array
                             does nothing if filtering is 'linear'.
        avg_coherent, boolean, if True integrate coherently.
        sq_units, if True, use square units (product of even/odd data FT).
        cache, dictionary containing filtering matrices.
        t_threshold, if fraction of flagged channels at single time is a above this, flag entire time.
        f_threshold, if fraction of flagged channels at a single freq is above this, flag entire freq at all times.
        fourier_taper, taper unfiltered boxcars in fourier domain (to get rid of residual side-lobes).
                       WARNING: This will help get rid of side-lobes of high domain contamination but
                       will mask low-delay structures!
        norm_zero_delay, if True, normalize each time by zero delay.
    '''
    if data_d is None:
        data, data_d = generate_sum_diff(data)


    if filter_method == 'linear':
        xg, yg, darray, darray_d = filter_data_linear(data = data ,data_d = data_d,corrkey = corrkey, fmin = fmin, fmax = fmax,
                             fringe_rate_max = fringe_rate_max, delay_max = delay_max, delay_center = delay_center,
                             lst_min = lst_min, lst_max=lst_max, taper=taper, fourier_taper = fourier_taper,
                             freq_domain = freq_domain, zero_flags = zero_flags, norm_zero_delay = norm_zero_delay,
                             time_domain = "time",tol=tol,f_threshold = f_threshold, bad_resid = bad_resid, bad_wghts = bad_wghts,
                             flag_across_time = flag_across_time,t_threshold = t_threshold,
                             fringe_rate_filter = fringe_rate_filter)

    elif filter_method == 'clean':
        xg, yg, darray, darray_d = filter_data_clean(data = data,data_d = data_d,corrkey = corrkey,fmin = fmin, fmax = fmax,
                             fringe_rate_max = fringe_rate_max, area_widths = delay_max, area_centers = delay_center, norm_zero_delay = norm_zero_delay,
                             lst_min = lst_min,lst_max=lst_max,taper=taper,filt2d_mode='rect',
                             add_clean_components=add_clean_components,freq_domain = freq_domain, bad_resid = bad_resid, bad_wghts = bad_wghts,
                             time_domain = "time",tol=tol, f_threshold = f_threshold,
                             flag_across_time = flag_across_time, t_threshold = t_threshold,
                             fringe_rate_filter = fringe_rate_filter,acr=False)
    else:
        raise ValueError("Failed to specify a valid filtering method. Valid options are 'clean' and 'linear'")
    #split data into even and odd sets.

    darray_even = (darray + darray_d) / 2.
    darray_odd = (darray - darray_d) / 2.
    darray_d_even =  darray_d[::2]
    darray_d_odd = darray_d[1::2]

    trace_o = np.zeros((darray.shape[0]/2, darray.shape[1]), dtype=complex)
    trace_e = np.zeros_like(trace_o)
    trace_e_d = np.zeros_like(trace_e)
    trace_o_d = np.zeros_like(trace_e)

    times = np.unique(times.time_array)
    ntimes = len(times)
    norm_vec = np.arange(1, ntimes/2)

    if avg_coherent:
        trace_e[0] = np.sum(darray_even[:2], axis = 0)
        trace_o[0] = np.sum(darray_odd[:2], axis = 0)
        trace_e_d[0] = np.sum(darray_d_even[:2], axis=0)
        trace_o_d[0] = np.sum(d_array_d_odd[:2], axis=0)
        for tind in range(1,ntimes/2):
            trace_e[tind] = trace_e[tind-1] + np.sum(darray_e[tind*2:(tind+1)*2], axis=0)
            trace_o[tind] = trace_o[tind-1] + np.sum(darray_o[tind*2:(tind+1)*2], axis=0)
            trace_e_d[tind] = trace_e_d[tind-1] + darray_d_even[tind]
            trace_o_d[tind] = trace_o_d[tind-1] + darray_d_odd[tind]
        trace = norm_vec ** -2. * trace_e * np.conj(trace_o) / 2.
        trace_d = norm_vec ** -2. * trace_e_d * np.conj(trac_o_d) / 4.

    else:
        trace[0] = np.sum(darray_even[:2] * np.conj(darray_odd[:2]) , axis = 0)
        trace_d[0] = darray_d_even[0] * np.conj(darray_d_odd[0])
        for tind in range(1, ntimes/2):
            trace[tind] = trace[tind-1] + np.sum(darray_even[tind*2:2*(tind+1)]\
            * np.conj(darray_odd[tind*2:(tind+1)*2]), axis = 0)
            trace_d[tind] = trace_d[tind-1] + darray_d_even[tind]\
            * np.conj(darray_d_odd[tind])

        trace = norm_vec ** -2. * trace / 2.
        trace_d = norm_vec ** -2. * trace_d / 2. / np.sqrt(2.)

    t = times[::2]
    x = xg[0,:]

    return t, x, trace, trace_d

def filter_and_average_abs(data, corrkey, data_d = None, fmin=45e6, fmax = 85e6, fringe_rate_max = .2e-3, delay_max = 300e-9, delay_center = 0.,
                           lst_min = None, lst_max = None, taper = 'boxcar', freq_domain = 'delay', zero_flags = True, normalize_average = False, lst_norm_max = None,
                           lst_norm_min = None, time_units = 'lst',
                           tol = 1e-7, flag_across_time = True, fringe_rate_filter = False, filter_method = 'linear', negative_vals = False,manual_flags = [],
                           manual_override_flags = False, add_clean_components = True, show_legend = True, avg_coherent = True, return_y = False,
                           sq_units = True, cache = WMAT_CACHE, norm_zero_delay = False, t_threshold = 0.1, f_threshold = 0.1, npts_avg = None, normalize_std = False,
                           fourier_taper = None, extra_chan_flags = [], positive_delay_only = False, bad_wghts = False, bad_resid=False):
    '''
    delay filter data and compute average.
    data, pyuvdata data set
    data_d, pyuvdata diffed data set, if None, generate automatically
    corrkey, 3-tuple or list (ant0, ant1, pol num) or list of 3-tuples (that will be averaged together)
    fmin, minimum frequency (Hz)
    fmax, maximum frequency (Hz)
    normalize_average, bool, if True, normalize data (and noise) by average of
                             absolute value of unflagged data.
    fringe_rate_max, filter all fringe-rates below this value (Hz)
    delay_max, filter all delays within this value's distance to delay_center (sec),
               can be provided as a list of floats to filter multiple delay windows.
    delay_center, filter all delays within delay_max of this delay. Can be provided as a list of floats to filter multiple dleay windows.
    lst_min, minimum lst to fringe-rate filter and average over.
    lst_max, maximum lst to fringe-rate filter and average over.
    taper, tapering function to apply during fft.
    freq_domain, output domain in frequency ("delay" or "frequency")
    zero_flags, if True, set data at flagged channels to zero before performing Fourier filter. If False, do not set flagged channels to zero
                and allow whatever is in these channels to be part of the data.
    tol, depth to clean too.
    flag_across_time, if True, flags at each frequency are the union of flags at that frequency across all times.
    fringe_rate_filter, if True, filter fringe rates with abs value below fringe_rate_max.
    filter_method, if 'linear', use linear WTL filter. if 'clean' perform 1d clean. This applies to both frequency and time domains.
    add_clean_components, if True, add 1d clean components back to clean residual. This will do nothing if filter_method is 'linear'.
    avg_coherent, if True, average data coherently.
                  if False, average data incoherently.
    sq_units: if True, take abs square of data
              if False, take square root of data
    manual_flags, list of 2-tuples with centers and widths of flagging windows.
    manual_override_flags, flags in manual_flags override pre-existing flags.
    cache: dictionary storing linear filter matrices. Ignored if filter_method = 'clean'
    t_threshold, if fraction of flagged channels at single time is a above this, flag entire time.
    f_threshold, if fraction of flagged channels at a single freq is above this, flag entire freq at all times.
    fourier_taper, taper unfiltered boxcars in fourier domain (to get rid of residual side-lobes).
                   WARNING: This will help get rid of side-lobes of high domain contamination but
                   will mask low-delay structures!
    norm_zero_delay, if true, normalize each delay by zero delay.
    bad_resid, if True use aipy resid style
    bad_wghts, if True, add blackmanharris to cleaning kernel.
    normalize_std, normalize each channel by STD of diff file.
    '''
    if not isinstance(corrkey, list):
        corrkey = [corrkey]
    try:
        x = delay_center[0][0]
    except:
        delay_center = [delay_center]

    try:
        x = delay_max[0][0]
    except:
        delay_max = [delay_max]
    darray_list = []
    darray_d_list = []
    if not lst_min is None:
        print('lst_min=%f'%lst_min)
    if not lst_max is None:
        print('lst_max=%f'%lst_max)
    for ckey,dc,dm in zip(corrkey,delay_center,delay_max):
        if data_d is None:
            data, data_d = generate_sum_diff(data)

        if filter_method == 'linear':
            xg, yg, darray, darray_d = filter_data_linear(data = data ,data_d = data_d,corrkey = ckey, fmin = fmin, fmax = fmax,
                                 fringe_rate_max = fringe_rate_max, delay_max = dm, delay_center = dc,
                                 lst_norm_min = lst_norm_min, lst_norm_max = lst_norm_max, time_units = time_units, 
                                 lst_min = lst_min, lst_max=lst_max, taper=taper, normalize_average = normalize_average,
                                 manual_override_flags = manual_override_flags, manual_flags = manual_flags,
                                 freq_domain = freq_domain, zero_flags = zero_flags, fourier_taper = fourier_taper,
                                 time_domain = "time",tol=tol, t_threshold = t_threshold, extra_chan_flags = extra_chan_flags,
                                 flag_across_time = flag_across_time, f_threshold = f_threshold, norm_zero_delay = norm_zero_delay,
                                 fringe_rate_filter = fringe_rate_filter, positive_delay_only = positive_delay_only)

        elif filter_method == 'clean':
            xg, yg, darray, darray_d = filter_data_clean(data = data,data_d = data_d,corrkey = ckey,fmin = fmin, fmax = fmax,
                                 fringe_rate_max = fringe_rate_max, area_widths = dm, area_centers = dc, norm_zero_delay = norm_zero_delay,
                                 lst_min = lst_min,lst_max=lst_max,taper=taper,filt2d_mode='rect', extra_chan_flags = extra_chan_flags,
                                 lst_norm_min = lst_norm_min, lst_norm_max = lst_norm_max,
                                 add_clean_components=add_clean_components,freq_domain = freq_domain,
                                 manual_override_flags = manual_override_flags, manual_flags = manual_flags,
                                 time_domain = "time",tol=tol,bad_wghts=bad_wghts,normalize_average = normalize_average,
                                 flag_across_time = flag_across_time,bad_resid=bad_resid, t_threshold = t_threshold,
                                 fringe_rate_filter = fringe_rate_filter,acr=False, f_threshold = f_threshold, positive_delay_only = positive_delay_only)
        else:
            raise ValueError("Failed to specify a valid filtering method. Valid options are 'clean' and 'linear'")
        #split data into even and odd sets.
        print('maximum time=%f'%yg.squeeze().max())
        print('maximum freq=%f'%xg.squeeze().max())
        if npts_avg is None:
            npts_avg = darray.shape[0]

        darray_even = (darray + darray_d) / 2.
        darray_odd = (darray - darray_d) / 2.
        darray_d_even =  darray_d[::2]
        darray_d_odd = darray_d[1::2]

        #print(np.any(np.isnan(darray_even)))
        #print(np.any(np.isnan(darray_even)))

        output_even = np.zeros((int(darray_even.shape[0]/npts_avg),darray_even.shape[1]) ,dtype = complex)
        output_odd =  np.zeros_like(output_even)
        output_d_even = np.zeros_like(output_even)
        output_d_odd = np.zeros_like(output_d_even)
        npts_avg_d = npts_avg // 2
        output = np.zeros_like(output_even)
        output_d = np.zeros_like(output_d_even)
        if avg_coherent:
            for m in range(output_even.shape[0]):
                output_even[m] = np.mean(darray_even[m*npts_avg:(m+1)*npts_avg], axis = 0)
                output_odd[m] = np.mean(darray_odd[m*npts_avg:(m+1)*npts_avg], axis = 0)
            for m in range(output_d_even.shape[0]):
                output_d_even[m] = np.mean(darray_d_even[m*npts_avg_d:(m+1)*npts_avg_d], axis = 0)
                output_d_odd[m] = np.mean(darray_d_odd[m*npts_avg_d:(m+1)*npts_avg_d], axis = 0)

            #darray_d_even = np.mean(darray_d_even, axis = 0)
            #darray_d_odd = np.mean(darray_d_odd, axis = 0)

            output_d = output_d_even * np.conj(output_d_odd ) / 4.
            output = output_even * np.conj(output_odd)

        if not avg_coherent:
            darray = darray_even * np.conj(darray_odd)
            darray_d = darray_d_even * np.conj(darray_d_odd)

            for m in range(output.shape[0]):
                output[m] = np.mean(darray[m*npts_avg:(m+1)*npts_avg], axis = 0)
            for m in range(output_d.shape[0]):
                output_d[m] = np.mean(darray_d[m*npts_avg:(m+1)*npts_avg], axis = 0) * np.sqrt(2.)

            #output_d = np.mean(output_d, axis = 0) * np.sqrt(2.)
    #print('output shape =')
    #print(output.shape)
    #print('output_d shape = ')
    #print(output_d.shape)
    darray_list.append(output)
    darray_d_list.append(output_d)

    darray = np.mean(np.asarray(darray_list),axis=0)
    darray_d = np.mean(np.asarray(darray_d_list),axis=0)

    if not sq_units:
        darray = sqrt_abs(darray,negatives = negative_vals)
        darray_d = sqrt_abs(darray_d, negatives = negative_vals)
    nan_std = ~np.isnan(darray_d)
    if normalize_std:
        for cnum in range(darray.shape[0]):
            nfactor = np.sqrt(np.mean(np.abs(darray_d[nan_std[:,cnum],cnum])**2.))
            darray[:,cnum] /= nfactor
            darray_d[:,cnum] /= nfactor

    if not return_y:
        return xg[0,:].squeeze(), darray.squeeze(), darray_d.squeeze()
    else:
        return xg[0,:].squeeze(), yg[::npts_avg,0].squeeze(), darray.squeeze(), darray_d.squeeze()

def waterfall_plot(plot_dict, sq_units = True, freq_domain = 'delay', ylim = (None,None), show_k = False, delay_step = None, npts_avg = 1, normalize_std=False,
                   xlim = (None,None), logscale = True, label_font_size = 14, tick_font_size = 14, title = None, freq_units = 'MHz', title_y = 1.,lst_norm_min = None,
                   positive_delay_only = False, time_domain = 'time', lstmin = None, lstmax = None, negative_vals = True, title_font_size = 20, time_units = 'lst', lst_norm_max = None):
           '''
           DATA, a pyuvdata object containing primary data.
           DATA_DIFF, a pyuvdata object containing diffed data.
           CORRKEY (ant0, ant1, pol)
           LINESTYLE, linestyle to use
           COLOR, color of line
           NORMALIZE_AVERAGE, if True, divide data (and diff data) by data average.
           LINEWIDTH, width of line
           FMIN, minimum frequency
           FMAX, maximum frequency
           DELAY_CENTERS, float delay center or list of delay centers (for multiple windows)
           DELAY_WIDTHS, float delay width or list of delay widths
           FRINGE_RATE_FILTER, boolean. If True, apply fringe rate filter
           FRINGE_RATE_MAX, float, specifies the maximum fringe-rate to filter out.
           AVG_COHERENT, boolean, specifies whether a coherent or incoherent average should be taken.
           ADD_CLEAN_MODEL, boolean, specifies whether a clean model should be added.
           LST_MIN, minimum LST to include in data averaging. can be None
           LST_MAX, maximum LST to include in data averagine. can be None
           FILTER_METHOD, string specifying "clean" or "linear" filtering
           ZERO_FLAGS: boolena, specifying whether flagged channels should be zeroed out
           CACHE, optional argument that lets user input cache of weighting matrices.
                  a cache for linear filtering matrices.
           FLAG_ACROSS_TIME, boolean, if True, each frequency flag is set by the union of all
                            flags at that frequency.
           LABEL, string, a label for the line.
           SHOW_HORIZON, if True, show verticale lines at baseline horizon
           SHOW_FILTER, if True, show vertical lines at filter edges.
           TOL, tolerance to clean/filter too.
           TAPER, string giving taper for FT.
           MANUAL_OVERRIDE_FLAGS,flags in manual_flags override pre-existing flags.
           MANUAL_FLAGS,list of 2-tuples with centers and widths of flagging windows.
           NORMALIZE_ZERO_DELAY, if true, normalize each time separately to zero delay value.
           T_THRESHOLD, if fraction of flagged channels at single time is a above this, flag entire time.
           F_THRESHOLD, if fraction of flagged channels at a single freq is above this, flag entire freq at all times.
           FOURIER_TAPER, taper unfiltered boxcars in fourier domain (to get rid of residual side-lobes).
                          WARNING: This will help get rid of side-lobes of high domain contamination but
                          will mask low-delay structures!
           BAD_RESID: use aipy residual (which I think is bad -- subjective nomenclature)
           BAD_WGHTS: use clean kernel that includes tapering function.
           normalize_std: nomralize by std of diff files.
           '''
           xlim = [xlim[0],xlim[1]]
           ylim = [ylim[0],ylim[1]]
           xlim_in = copy.copy(xlim)
           ylim_in = copy.copy(ylim)

           xlim = copy.copy(xlim)
           ylim = copy.copy(ylim)

           if ylim_in[1] is None:
               ylim[1] = -9e99
           if ylim_in[0] is None:
               ylim[0] = 9e99

           if xlim_in[1] is None:
               xlim[1] = -9e99
           if xlim_in[0] is None:
               xlim[0] = 9e99


           pd = plot_dict
           if lstmin is None:
                lstmin = pd['LST_MIN']
           if lstmax is None:
                lstmax = pd['LST_MAX']
           x0, x1, y, yd = filter_and_average_abs(data = pd['DATA'], data_d = pd['DATA_DIFF'],
                                   corrkey = pd['CORRKEY'], fmin = pd['FMIN'],lst_norm_min = lst_norm_min, lst_norm_max = lst_norm_max,
                                   fmax = pd['FMAX'], fringe_rate_max = pd['FRINGE_RATE_MAX'],
                                   delay_max=pd['DELAY_WIDTHS'], delay_center = pd['DELAY_CENTERS'],
                                   lst_min = lstmin, lst_max = lstmax,normalize_std = normalize_std,
                                   taper = pd['TAPER'], freq_domain = freq_domain,
                                   zero_flags = pd['ZERO_FLAGS'], tol = pd['TOL'], time_units = time_units, 
                                   flag_across_time = pd['FLAG_ACROSS_TIME'], bad_wghts = pd['BAD_WGHTS'], bad_resid = pd['BAD_RESID'],
                                   fringe_rate_filter = pd['FRINGE_RATE_FILTER'], manual_flags = pd['MANUAL_FLAGS'],
                                   filter_method = pd['FILTER_METHOD'], norm_zero_delay = pd['NORMALIZE_ZERO_DELAY'],
                                   add_clean_components = pd['ADD_CLEAN_MODEL'], manual_override_flags = pd['MANUAL_OVERRIDE_FLAGS'],
                                   avg_coherent = pd['AVG_COHERENT'], positive_delay_only = positive_delay_only,
                                   f_threshold = pd['F_THRESHOLD'], fourier_taper = pd['FOURIER_TAPER'],
                                   t_threshold = pd['T_THRESHOLD'], extra_chan_flags = pd['CHANNEL_FLAGS'],
                                   normalize_average = pd['NORMALIZE_AVERAGE'],npts_avg = npts_avg, return_y = True,
                                   sq_units = sq_units, cache = pd['CACHE'], negative_vals = negative_vals)
           print(yd.shape)
           if freq_domain == 'delay':
               x0 *= 1e9
           elif freq_domain == 'frequency':
               x0 *= 1e-6
           if str(freq_units).lower() == 'channel':
                x0 =(x0 - x0[0])/(x0[1]-x0[0])
           elif str(freq_units).lower() == 'ghz':
                x0 = x0*1e-3
           elif str(freq_units).lower() == 'hz':
                x0 = x0*1e6

           if negative_vals:
                y = np.real(y)
                yd = np.real(yd)
           else:
                y = np.abs(y)
                yd = np.abs(yd)

           cklist = pd['CORRKEY']
           if not isinstance(cklist,list):
                cklist = [cklist]

           if pd['SHOW_HORIZON'] and freq_domain == 'delay':
                for ckey in cklist:
                    hzn = get_horizon(pd['DATA'],(ckey[0],ckey[1]))
                    plt.axvline(hzn*1e9, ls = pd['LINESTYLE'], color = [.5,.5,.5])
                    plt.axvline(-hzn*1e9, ls = pd['LINESTYLE'], color = [.5,.5,.5])

           if ylim_in[0] is None:
               if np.abs(y).max() > ylim[1]:
                   ylim[1] = np.abs(y).max()
                   ylim[1] = 10.**np.ceil(np.log10(ylim[1])) * 10.

           if ylim_in[1] is None:
               if np.abs(yd).mean() < ylim[0]:
                   ylim[0] = np.abs(yd).mean()
                   ylim[0] = 10.**np.floor(np.log10(ylim[0])) / 10.


           if xlim_in[1] is None:
               if x.max() > xlim[1]:
                   xlim[1] = x.max()
           if xlim_in[0] is None:
               if x.min() < xlim[0]:
                   xlim[0] = x.min()

           x0g, x1g = np.meshgrid(x0, x1)


           if logscale:
            yd = np.log10(yd)
            y = np.log10(y)
            ylim[0] = np.log10(ylim[0])
            ylim[1] = np.log10(ylim[1])
           
           pc= plt.pcolor(x0g, x1g, y, vmin = ylim[0], vmax = ylim[1])
           nyticks = 10
           yticks = np.linspace(x1g.min(), x1g.max(), nyticks)
           if time_units == 'sast':
               utc_offset = 2.
           else:
               utc_offset = 0.
           tz = TimezoneInfo(utc_offset = utc_offset * units.hour)
           

           if freq_domain == 'delay':
               ax1=plt.gca()
               if show_k and not no_labels:
                   #plot k-parallel axis above plot if we are in the delay domain.
                   f0 = (pd['FMIN'] + pd['FMAX'])/2.
                   z0 = 1420.41e6/f0 - -1.
                   y0 = 3e5/100. * (1.+z0)**2. / np.sqrt(.7 + .3 * (1.+z0)**3.) / 1420.41e6
                   #delay_step = pd['DELAY_STEP']
                   if not (delay_step is None):
                       ax1.set_xticks(np.arange(xlim[0],xlim[1]+delay_step,delay_step))

                   ax2 = plt.gca().twiny()
                   ax2.set_xlim(ax1.get_xlim())
                   ax2.set_xticks(ax1.get_xticks())
                   ax2ticks = []
                   for tick in ax1.get_xticks():
                       kpara = tick * 2. * np.pi / y0 /1e9
                       ktick = '%.2f'%(kpara)
                       ax2ticks.append(ktick)
                   ax2.set_xticklabels(ax2ticks)
                   ax2.set_xlabel('$k_\\parallel$ ($h$Mpc$^{-1}$)', fontsize = label_font_size)
                   plt.sca(ax1)
               ax1.set_xlabel('$\\tau_d$ (ns)', fontsize = label_font_size)
               if sq_units:
                   plt.ylabel('|$\\widetilde{V}(\\tau)|^2$', fontsize = label_font_size)
               else:
                   plt.ylabel('|$\\widetilde{V}(\\tau)|$', fontsize = label_font_size)

           else:
               plt.xlabel('$\\nu$ (MHz)', fontsize = label_font_size)
               #if sq_units:
               #   plt.ylabel('|$V(\\nu)|^2$', fontsize = label_font_size)
               #else:
               #   plt.ylabel('|$V(\\nu)|$', fontsize = label_font_size)
               if time_units == 'lst':
                   plt.ylabel('LST (Hours)', fontsize = label_font_size)
               elif time_units == 'sast':
                   plt.ylabel('SAST', fontsize = label_font_size)
               elif time_units =='utc':
                   plt.ylabel('UTC', fontsize = label_font_size)
               elif time_units == 'jd':
                   plt.ylabel('JD', fontsize = label_font_size)
           plt.tick_params(labelsize = tick_font_size)
           if not title is None:
               plt.title(title,fontsize = title_font_size, y = title_y)
           #if time_units == 'jd':
           #    print(plt.gca().get_yticks())

           if time_units == 'sast' or time_units == 'utc':
               tick_labels = []
               for tick in yticks:
                   #print(tick)
                   tjd = Time(float(tick), format='jd')
                   tdt = tjd.to_datetime(timezone = tz)
                   tlabel = '%d %02d-%02d %02d:%02d'%(tdt.year, tdt.month, tdt.day, tdt.hour, tdt.minute)
                   #print(tlabel)
                   tick_labels.append(tlabel)
               plt.gca().set_yticks(yticks)
               plt.gca().set_yticklabels(tick_labels)
                   

           return pc




def avg_comparison_plot(plot_dict_list, sq_units = True,freq_domain = 'delay', ylim = (None, None),show_k=False,delay_step=None,lst_norm_min = None, lst_norm_max = None,
                                xlim = (None,None),logscale = True, legend_font_size = 14, show_signal = True, show_diff = True, lstmin=None, lstmax=None,
                                label_font_size = 14, tick_font_size = 14, legend_loc = 'lower center', title = None,alpha=1.,alpha_diff = .25, ls = None,
                                title_font_size = 18, title_y = 1.1, no_labels = False, freq_units = 'MHz', positive_delay_only = False, color_override = None,
                                legend_ncol = None, legend_bbox = (0.5, 0.), show_legend = True, negative_vals = False,lw_override=None, ls_override = None,
                                show_filter_override = False):
    '''
    plot_dict_list: a list of dictionaries specifying the plotting parameters of each line.
    each dictionary must have the following:
        DATA, a pyuvdata object containing primary data.
        DATA_DIFF, a pyuvdata object containing diffed data.
        CORRKEY (ant0, ant1, pol)
        LINESTYLE, linestyle to use
        COLOR, color of line
        NORMALIZE_AVERAGE, if True, divide data (and diff data) by data average.
        LINEWIDTH, width of line
        FMIN, minimum frequency
        FMAX, maximum frequency
        DELAY_CENTERS, float delay center or list of delay centers (for multiple windows)
        DELAY_WIDTHS, float delay width or list of delay widths
        FRINGE_RATE_FILTER, boolean. If True, apply fringe rate filter
        FRINGE_RATE_MAX, float, specifies the maximum fringe-rate to filter out.
        AVG_COHERENT, boolean, specifies whether a coherent or incoherent average should be taken.
        ADD_CLEAN_MODEL, boolean, specifies whether a clean model should be added.
        LST_MIN, minimum LST to include in data averaging. can be None
        LST_MAX, maximum LST to include in data averagine. can be None
        FILTER_METHOD, string specifying "clean" or "linear" filtering
        ZERO_FLAGS: boolena, specifying whether flagged channels should be zeroed out
        CACHE, optional argument that lets user input cache of weighting matrices.
               a cache for linear filtering matrices.
        FLAG_ACROSS_TIME, boolean, if True, each frequency flag is set by the union of all
                         flags at that frequency.
        LABEL, string, a label for the line.
        SHOW_HORIZON, if True, show verticale lines at baseline horizon
        SHOW_FILTER, if True, show vertical lines at filter edges.
        TOL, tolerance to clean/filter too.
        TAPER, string giving taper for FT.
        MANUAL_OVERRIDE_FLAGS,flags in manual_flags override pre-existing flags.
        MANUAL_FLAGS,list of 2-tuples with centers and widths of flagging windows.
        NORMALIZE_ZERO_DELAY, if true, normalize each time separately to zero delay value.
        T_THRESHOLD, if fraction of flagged channels at single time is a above this, flag entire time.
        F_THRESHOLD, if fraction of flagged channels at a single freq is above this, flag entire freq at all times.
        FOURIER_TAPER, taper unfiltered boxcars in fourier domain (to get rid of residual side-lobes).
                       WARNING: This will help get rid of side-lobes of high domain contamination but
                       will mask low-delay structures!
        BAD_RESID: use aipy residual (which I think is bad -- subjective nomenclature)
        BAD_WGHTS: use clean kernel that includes tapering function.
    show_signal, if True, plot signal data
    show_diff, if True, plot diff data.
    freq_domain, string, specify if output is in "frequency" or "delay" domain.
    ylim, 2-tuple with upper and lower bounds on plot. If bound is None, will be rounded
          to nearest order of magnitude.
    xlim, 2-tuple with upper and lower x-limits on plot.
    sq_units, if True, show the square of the delay-transform. If false, show the
             square root of the absoute value.
    logscale, if True, y-axis is logarithmically scaled.
    negative_vals, if True, let negative numbers be negative.
    title, string for plot title
    title_font_size, float, font size of title
    title_y, float, y location of title.
    no_labels, if True, show no labels (including ticks). You might want to
            set true if you will be adding additinal lines to same axis.
    lw_override, specify an override line width that is used instead of LINE_WIDTH. Default, None.
    Returns:
        lines, labels, figure handle, axis handle.
    '''
    xlim = [xlim[0],xlim[1]]
    ylim = [ylim[0],ylim[1]]
    xlim_in = copy.copy(xlim)
    ylim_in = copy.copy(ylim)

    xlim = copy.copy(xlim)
    ylim = copy.copy(ylim)

    if ylim_in[1] is None:
        ylim[1] = -9e99
    if ylim_in[0] is None:
        ylim[0] = 9e99

    if xlim_in[1] is None:
        xlim[1] = -9e99
    if xlim_in[0] is None:
        xlim[0] = 9e99

    lines = []
    labels = []
    
    values = []

    for pd in plot_dict_list:
        if lstmin is None:
            lstmin = pd['LST_MIN']
        if lstmax is None:
            lstmax = pd['LST_MAX']
        x, y, yd = filter_and_average_abs(data = pd['DATA'], data_d = pd['DATA_DIFF'],
                                corrkey = pd['CORRKEY'], fmin = pd['FMIN'],
                                fmax = pd['FMAX'], fringe_rate_max = pd['FRINGE_RATE_MAX'],
                                delay_max=pd['DELAY_WIDTHS'], delay_center = pd['DELAY_CENTERS'],
                                lst_min = lstmin, lst_max = lstmax, lst_norm_min = lst_norm_min, lst_norm_max = lst_norm_max,
                                taper = pd['TAPER'], freq_domain = freq_domain,
                                zero_flags = pd['ZERO_FLAGS'], tol = pd['TOL'],
                                flag_across_time = pd['FLAG_ACROSS_TIME'], bad_wghts = pd['BAD_WGHTS'], bad_resid = pd['BAD_RESID'],
                                fringe_rate_filter = pd['FRINGE_RATE_FILTER'], manual_flags = pd['MANUAL_FLAGS'],
                                filter_method = pd['FILTER_METHOD'], norm_zero_delay = pd['NORMALIZE_ZERO_DELAY'],
                                add_clean_components = pd['ADD_CLEAN_MODEL'], manual_override_flags = pd['MANUAL_OVERRIDE_FLAGS'],
                                avg_coherent = pd['AVG_COHERENT'], positive_delay_only = positive_delay_only,
                                f_threshold = pd['F_THRESHOLD'], fourier_taper = pd['FOURIER_TAPER'],
                                t_threshold = pd['T_THRESHOLD'], extra_chan_flags = pd['CHANNEL_FLAGS'],
                                normalize_average = pd['NORMALIZE_AVERAGE'],
                                sq_units = sq_units, cache = pd['CACHE'], negative_vals = negative_vals)

        #print(yd.shape)
        #print(y.shape)
        #print(x.shape)
        
        values.append([x,y,yd])

        if freq_domain == 'delay':
            x *= 1e9
        elif freq_domain == 'frequency':
            x *= 1e-6


        if str(freq_units).lower() == 'channel':
            x =(x - x[0])/(x[1]-x[0])
        elif str(freq_units).lower() == 'ghz':
            x = x*1e-3
        elif str(freq_units).lower() == 'hz':
            x = x*1e6

        if negative_vals:
            y = np.real(y)
            yd = np.real(yd)
        else:
            y = np.abs(y)
            yd = np.abs(yd)

        if ls_override is None:
            ls = pd['LINESTYLE']
        else:
            ls = ls_override

        if color_override is None:
            color = pd['COLOR']
        else:
            color = color_override

        if lw_override is None:
            lw = pd['LINEWIDTH']
        else:
            lw = lw_override

        if show_signal:
            lines.append(plt.plot(x,y,lw=lw,
                              color=color,
                              ls=ls,alpha=alpha)[0])
            if show_diff:
                plt.plot(x,yd,lw=1,ls=ls,
                     color=color,alpha=alpha_diff)


        elif show_diff:
            lines.append(plt.plot(x,yd,lw=1,ls=pd['LINESTYLE'],
                 color=pd['COLOR'])[0])

        labels.append(pd['LABEL'])
        cklist = pd['CORRKEY']
        if not isinstance(cklist,list):
            cklist = [cklist]

        if pd['SHOW_HORIZON'] and freq_domain == 'delay':
            for ckey in cklist:
                hzn = get_horizon(pd['DATA'],(ckey[0],ckey[1]))
                plt.axvline(hzn*1e9, ls = pd['LINESTYLE'], color = [.5,.5,.5])
                plt.axvline(-hzn*1e9, ls = pd['LINESTYLE'], color = [.5,.5,.5])



        if ylim_in[0] is None:
            if np.abs(y).max() > ylim[1]:
                ylim[1] = np.abs(y).max()
                ylim[1] = 10.**np.ceil(np.log10(ylim[1])) * 10.

        if ylim_in[1] is None:
            if np.abs(yd).mean() < ylim[0]:
                ylim[0] = np.abs(yd).mean()
                ylim[0] = 10.**np.floor(np.log10(ylim[0])) / 10.


        if xlim_in[1] is None:
            if x.max() > xlim[1]:
                xlim[1] = x.max()
        if xlim_in[0] is None:
            if x.min() < xlim[0]:
                xlim[0] = x.min()


        dclist = pd['DELAY_CENTERS']
        dwlist = pd['DELAY_WIDTHS']

        if not isinstance(dclist[0], list):
            dclist = [dclist]
        if not isinstance(dwlist[0], list):
            dwlist = [dwlist]

        if show_filter_override is None:
            show_filter = pd['SHOW_FILTER']
        else:
            show_filter = show_filter_override

        if show_filter and freq_domain == 'delay':
            for dcl,dwl in zip(dclist,dwlist):
                for dc,dw in zip(dcl,dwl):
                    plt.fill_between(np.array([dc-dw,dc+dw])*1e9,[ylim[0],ylim[0]],[ylim[1],ylim[1]],color=pd['COLOR'],alpha=.1)
                    #plt.axvline((dc+dw)*1e9, ls = '-.', color = [.75,.75,.75])
                    #plt.axvline((dc-dw)*1e9, ls = '-.', color = [.75,.75,.75])
        if show_filter and freq_domain == 'frequency':
            for fc,fw in pd['MANUAL_FLAGS']:
                plt.fill_between(np.array([fc-fw,fc+fw])/1e6,[ylim[0],ylim[0]],[ylim[1],ylim[1]],color=pd['COLOR'],alpha=.1)


    if logscale:
        plt.yscale('log')

    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    if freq_domain == 'delay':
        ax1=plt.gca()
        if show_k and not no_labels:
            #plot k-parallel axis above plot if we are in the delay domain.
            f0 = (pd['FMIN'] + pd['FMAX'])/2.
            z0 = 1420.41e6/f0 - -1.
            y0 = 3e5/100. * (1.+z0)**2. / np.sqrt(.7 + .3 * (1.+z0)**3.) / 1420.41e6
            #delay_step = pd['DELAY_STEP']
            if not (delay_step is None):
                ax1.set_xticks(np.arange(xlim[0],xlim[1]+delay_step,delay_step))

            ax2 = plt.gca().twiny()
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticks(ax1.get_xticks())
            ax2ticks = []
            for tick in ax1.get_xticks():
                kpara = tick * 2. * np.pi / y0 /1e9
                ktick = '%.2f'%(kpara)
                ax2ticks.append(ktick)
            ax2.set_xticklabels(ax2ticks)
            ax2.set_xlabel('$k_\\parallel$ ($h$Mpc$^{-1}$)', fontsize = label_font_size)
            plt.sca(ax1)
        ax1.set_xlabel('$\\tau_d$ (ns)', fontsize = label_font_size)
        if sq_units:
            plt.ylabel('|$\\widetilde{V}(\\tau)|^2$', fontsize = label_font_size)
        else:
            plt.ylabel('|$\\widetilde{V}(\\tau)|$', fontsize = label_font_size)

    else:
        plt.xlabel('$\\nu$ (MHz)', fontsize = label_font_size)
        if sq_units:
            plt.ylabel('|$V(\\nu)|^2$', fontsize = label_font_size)
        else:
            plt.ylabel('|$V(\\nu)|$', fontsize = label_font_size)
    plt.tick_params(labelsize = tick_font_size)
    if legend_ncol is None:
        legend_ncol = len(lines)
    if show_legend:
        plt.gcf().legend(lines, labels, loc=legend_loc, ncol = legend_ncol,
                        fontsize=legend_font_size, bbox_to_anchor=legend_bbox)
    if not title is None:
        plt.title(title,fontsize = title_font_size, y = title_y)
    #if no_labels:
    #    plt.gca().set_xticklabels([])
    #    plt.gca().set_yticklabels([])
    #    plt.gca().set_xlabel('')
    #    plt.gca().set_ylabel('')
    plt.grid()
    return lines, labels, plt.gcf(), plt.gca(), values



def time_comparison_plot(plot_dict_list, sq_units = True,freq_domain = 'delay', ylim = [None, None],
                                xlim = [None,None],logscale = True, legend_font_size = 14, show_signal = True, show_diff = True,
                                label_font_size = 14, tick_font_size = 14, legend_loc = 'lower center', title = None,
                                title_font_size = 18, title_y = 1.1, no_labels = False, freq_units = 'MHz', positive_delay_only = False,
                                legend_ncol = None, legend_bbox = [0.5, 0.], show_legend = True, negative_vals = False):
    '''
    plot_dict_list: a list of dictionaries specifying the plotting parameters of each line.
    each dictionary must have the following:
        DATA, a pyuvdata object containing primary data.
        DATA_DIFF, a pyuvdata object containing diffed data.
        CORRKEY (ant0, ant1, pol)
        LINESTYLE, linestyle to use
        COLOR, color of line
        NORMALIZE_AVERAGE, if True, divide data (and diff data) by data average.
        LINEWIDTH, width of line
        FMIN, minimum frequency
        FMAX, maximum frequency
        DELAY_CENTERS, float delay center or list of delay centers (for multiple windows)
        DELAY_WIDTHS, float delay width or list of delay widths
        FRINGE_RATE_FILTER, boolean. If True, apply fringe rate filter
        FRINGE_RATE_MAX, float, specifies the maximum fringe-rate to filter out.
        AVG_COHERENT, boolean, specifies whether a coherent or incoherent average should be taken.
        ADD_CLEAN_MODEL, boolean, specifies whether a clean model should be added.
        LST_MIN, minimum LST to include in data averaging. can be None
        LST_MAX, maximum LST to include in data averagine. can be None
        FILTER_METHOD, string specifying "clean" or "linear" filtering
        ZERO_FLAGS: boolena, specifying whether flagged channels should be zeroed out
        CACHE, optional argument that lets user input cache of weighting matrices.
               a cache for linear filtering matrices.
        FLAG_ACROSS_TIME, boolean, if True, each frequency flag is set by the union of all
                         flags at that frequency.
        LABEL, string, a label for the line.
        SHOW_HORIZON, if True, show verticale lines at baseline horizon
        SHOW_FILTER, if True, show vertical lines at filter edges.
        TOL, tolerance to clean/filter too.
        TAPER, string giving taper for FT.
        T_THRESHOLD, if fraction of flagged channels at single time is a above this, flag entire time.
        F_THRESHOLD, if fraction of flagged channels at a single freq is above this, flag entire freq at all times.
        NORMALIZE_ZERO_DELAY, if true, normalize each time separately to zero delay value.
        MANUAL_OVERRIDE_FLAGS,flags in manual_flags override pre-existing flags.
        MANUAL_FLAGS,list of 2-tuples with centers and widths of flagging windows.
        BAD_RESID: use aipy residual (which I think is bad -- subjective nomenclature)
        BAD_WGHTS: use clean kernel that includes tapering function.
    show_signal, if True, plot signal data
    show_diff, if True, plot diff data.
    freq_domain, string, specify if output is in "frequency" or "delay" domain.
    ylim, 2-tuple with upper and lower bounds on plot. If bound is None, will be rounded
          to nearest order of magnitude.
    xlim, 2-tuple with upper and lower x-limits on plot.
    sq_units, if True, show the square of the delay-transform. If false, show the
             square root of the absoute value.
    logscale, if True, y-axis is logarithmically scaled.
    negative_vals, if True, let negative numbers be negative.
    title, string for plot title
    title_font_size, float, font size of title
    title_y, float, y location of title.
    no_labels, if True, show no labels (including ticks). You might want to
                set true if you will be adding additinal lines to same axis.
    Returns:
        lines, labels, figure handle, axis handle.
    '''
    xlim_in = copy.copy(xlim)
    ylim_in = copy.copy(ylim)

    xlim = copy.copy(xlim)
    ylim = copy.copy(ylim)

    if ylim_in[1] is None:
        ylim[1] = -9e99
    if ylim_in[0] is None:
        ylim[0] = 9e99

    if xlim_in[1] is None:
        xlim[1] = -9e99
    if xlim_in[0] is None:
        xlim[0] = 9e99

    lines = []
    labels = []

    for pd in plot_dict_list:
        if pd['FILTER_METHOD'] == 'linear':
            xg, yg, y, yd = filter_data_linear(data = pd['DATA'], data_d = pd['DATA_DIFF'],
                                    corrkey = pd['CORRKEY'], fmin = pd['FMIN'],
                                    fmax = pd['FMAX'], fringe_rate_max = pd['FRINGE_RATE_MAX'],
                                    delay_max=pd['DELAY_WIDTHS'], delay_center = pd['DELAY_CENTERS'],
                                    lst_min = pd['LST_MIN'], lst_max = pd['LST_MAX'],
                                    taper = pd['TAPER'], freq_domain = freq_domain,manual_override_flags = pd['MANUAL_OVERRIDE_FLAGS'],
                                    zero_flags = pd['ZERO_FLAGS'], tol = pd['TOL'], manual_flags = pd['MANUAL_FLAGS'],
                                    flag_across_time = pd['FLAG_ACROSS_TIME'], positive_delay_only = positive_delay_only,
                                    fringe_rate_filter = pd['FRINGE_RATE_FILTER'], norm_zero_delay = pd['NORMALIZE_ZERO_DELAY'],
                                    f_threshold = pd['F_THRESHOLD'], extra_chan_flags = pd['CHANNEL_FLAGS'],
                                    t_threshold = pd['T_THRESHOLD'], fourier_taper = pd['FOURIER_TAPER'],
                                    normalize_average = pd['NORMALIZE_AVERAGE'],
                                    cache = pd['CACHE'])
        elif pd['FILTER_METHOD'] == 'clean':
            xg, yg, y, yd = filter_data_clean(data = pd['DATA'], data_d = pd['DATA_DIFF'],
                                    corrkey = pd['CORRKEY'], fmin = pd['FMIN'],
                                    fmax = pd['FMAX'], fringe_rate_max = pd['FRINGE_RATE_MAX'],
                                    delay_max=pd['DELAY_WIDTHS'], delay_center = pd['DELAY_CENTERS'],
                                    lst_min = pd['LST_MIN'], lst_max = pd['LST_MAX'],
                                    taper = pd['TAPER'], freq_domain = freq_domain, bad_wghts = pd['BAD_WGHTS'],
                                    zero_flags = pd['ZERO_FLAGS'], tol = pd['TOL'], bad_resid = pd['BAD_RESID'],
                                    flag_across_time = pd['FLAG_ACROSS_TIME'], manual_flags = pd['MANUAL_FLAGS'],
                                    fringe_rate_filter = pd['FRINGE_RATE_FILTER'], manual_override_flags = pd['MANUAL_OVERRIDE_FLAGS'],
                                    add_clean_components = pd['ADD_CLEAN_MODEL'], norm_zero_delay = pd['NORMALIZE_ZERO_DELAY'],
                                    f_threshold = pd['F_THRESHOLD'], positive_delay_only = positive_delay_only,
                                    t_threshold = pd['T_THRESHOLD'], extra_chan_flags = pd['CHANNEL_FLAGS'],
                                    normalize_average = pd['NORMALIZE_AVERAGE'],
                                    cache = pd['CACHE'])

        #print(y)
        #print(yd)
        if negative_vals:
            y = np.real(y)
            yd = np.real(yd)
        else:
            y = np.abs(y)
            yd = np.abs(yd)
        x = xg[0,:]
        if freq_domain == 'delay':
            x *= 1e9
        elif freq_domain == 'frequency':
            x *= 1e-6
        if str(freq_units).lower() == 'channel':
            x =(x - x[0])/(x[1]-x[0])
        elif str(freq_units).lower() == 'ghz':
            x = x*1e-3
        elif str(freq_units).lower() == 'hz':
            x = x*1e6

        if negative_vals:
            y = np.real(y)
            yd = np.real(yd)
        else:
            y = np.abs(y)
            yd = np.abs(yd)
        if show_signal:
            lines.append(plt.plot(x,y[0],lw=pd['LINEWIDTH'],
                              color=pd['COLOR'],
                              ls=pd['LINESTYLE'])[0])
            for tnum in range(1,len(y)):
                plt.plot(x,y[tnum],lw=pd['LINEWIDTH'],
                                  color=pd['COLOR'],
                                  ls=pd['LINESTYLE'])
                if show_diff:
                    plt.plot(x,yd[tnum],lw=1,ls=pd['LINESTYLE'],
                         color=pd['COLOR'])


        elif show_diff:
            lines.append(plt.plot(x,yd[0],lw=1,ls=pd['LINESTYLE'],
                 color=pd['COLOR'])[0])
            for tnum in range(1,len(y)):
                plt.plot(x,yd[0],lw=1,ls=pd['LINESTYLE'],
                     color=pd['COLOR'])

        labels.append(pd['LABEL'])
        if pd['SHOW_HORIZON'] and freq_domain == 'delay':
            hzn = get_horizon(pd['DATA'],(pd['CORRKEY'][0],pd['CORRKEY'][1]))
            plt.axvline(hzn*1e9, ls = pd['LINESTYLE'], color = [.5,.5,.5])
            plt.axvline(-hzn*1e9, ls = pd['LINESTYLE'], color = [.5,.5,.5])


        if ylim_in[0] is None:
            if np.abs(y.flatten()).max() > ylim[1]:
                ylim[1] = np.abs(y.flatten()).max()
                ylim[1] = 10.**np.ceil(np.log10(ylim[1])) * 10.

        if ylim_in[1] is None:
            if np.abs(yd.flatten()).mean() < ylim[0]:
                ylim[0] = np.abs(yd.flatten()).mean()
                ylim[0] = 10.**np.floor(np.log10(ylim[0])) / 10.


        if xlim_in[1] is None:
            if x.max() > xlim[1]:
                xlim[1] = x.max()
        if xlim_in[0] is None:
            if x.min() < xlim[0]:
                xlim[0] = x.min()

        if pd['SHOW_FILTER'] and freq_domain == 'delay':
            if isinstance(pd['DELAY_WIDTHS'],float):
                pd['DELAY_WIDTHS'] = [pd['DELAY_WIDTHS']]
            if isinstance(pd['DELAY_CENTERS'],float):
                pd['DELAY_CENTERS'] = [pd['DELAY_CENTERS']]
            for dc,dw in zip(pd['DELAY_CENTERS'],pd['DELAY_WIDTHS']):
                plt.fill_between(np.array([dc-dw,dc+dw])*1e9,[ylim[0],ylim[0]],[ylim[1],ylim[1]],color=pd['COLOR'],alpha=.1)
                    #plt.axvline((dc+dw)*1e9, ls = '-.', color = [.75,.75,.75])
                    #plt.axvline((dc-dw)*1e9, ls = '-.', color = [.75,.75,.75])
        if pd['SHOW_FILTER'] and freq_domain == 'frequency':
            for fc,fw in pd['MANUAL_FLAGS']:
                plt.fill_between(np.array([fc-fw,fc+fw])/1e6,[ylim[0],ylim[0]],[ylim[1],ylim[1]],color=pd['COLOR'],alpha=.1)


    if logscale:
        plt.yscale('log')

    if freq_domain == 'delay':
        ax1=plt.gca()
        if pd['SHOW_K'] and not no_labels:
            #plot k-parallel axis above plot if we are in the delay domain.
            f0 = (pd['FMIN'] + pd['FMAX'])/2.
            z0 = 1420.41e6/f0 - -1.
            y0 = 3e5/100. * (1.+z0)**2. / np.sqrt(.7 + .3 * (1.+z0)**3.) / 1420.41e6
            ax1.set_xlim(xlim)
            plt.grid()
            delay_step = pd['DELAY_STEP']
            if not delay_step is None:
                ax1.set_xticks(np.arange(xlim[0],xlim[1]+delay_step,delay_step))
            ax2 = plt.gca().twiny()
            ax2.set_xticks(ax1.get_xticks())
            ax2.set_xlim(ax1.get_xlim())
            ax2ticks = []
            for tick in ax1.get_xticks():
                kpara = tick * 2. * np.pi / y0 /1e9
                ktick = '%.2f'%(kpara)
                ax2ticks.append(ktick)
            ax2.set_xticklabels(ax2ticks)
            ax2.set_xlabel('$k_\\parallel$ ($h$Mpc$^{-1}$)', fontsize = label_font_size)
            plt.sca(ax1)
        ax1.set_xlabel('$\\tau_d$ (ns)', fontsize = label_font_size)
        if sq_units:
            plt.ylabel('|$\\widetilde{V}(\\tau)|^2$', fontsize = label_font_size)
        else:
            plt.ylabel('|$\\widetilde{V}(\\tau)|$', fontsize = label_font_size)

    else:
        plt.xlabel('$\\nu$ (%s)'%freq_units, fontsize = label_font_size)
        if sq_units:
            plt.ylabel('|$V(\\nu)|^2$', fontsize = label_font_size)
        else:
            plt.ylabel('|$V(\\nu)|$', fontsize = label_font_size)
    plt.tick_params(labelsize = tick_font_size)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if legend_ncol is None:
        legend_ncol = len(lines)
    if show_legend:
        plt.gcf().legend(lines, labels, loc=legend_loc, ncol = legend_ncol,
                        fontsize=legend_font_size, bbox_to_anchor=legend_bbox)
    if not title is None:
        plt.title(title,fontsize = title_font_size, y = title_y)
    #if no_labels:
    #    plt.gca().set_xticklabels([])
    #    plt.gca().set_yticklabels([])
    #    plt.gca().set_xlabel('')
    #    plt.gca().set_ylabel('')
    return lines, labels, plt.gcf(), plt.gca()
