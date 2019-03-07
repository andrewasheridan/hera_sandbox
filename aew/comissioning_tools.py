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
    w = w / np.sqrt(np.mean(w**2.))
    return w

def sqrt_abs(x):
    '''
    sqrut of absolute value of x
    '''
    return np.sqrt(np.abs(x))


def high_pass_fourier_filter(data, wgts, filter_size, real_delta, clean2d=False, tol=1e-9, window='none',
                             skip_wgt=0.1, maxiter=100, gain=0.1, filt2d_mode='rect', alpha=0.5,
                             edgecut_low=0, edgecut_hi=0, add_clean_residual=False,bad_resid = False, bad_wghts = False):
    '''Apply a highpass fourier filter to data. Uses aipy.deconv.clean. Default is a 1D clean
    on the last axis of data.
    Arguments:
        data: 1D or 2D (real or complex) numpy array to be filtered.
            (Unlike previous versions, it is NOT assumed that weights have already been multiplied
            into the data.)
        wgts: real numpy array of linear multiplicative weights with the same shape as the data.
        filter_size: the half-width (i.e. the width of the positive part) of the region in fourier
            space, symmetric about 0, that is filtered out. In units of 1/[real_delta].
            Alternatively, can be fed as len-2 tuple specifying the absolute value of the negative
            and positive bound of the filter in fourier space respectively.
            Example: (20, 40) --> (-20 < tau < 40)
         real_delta: the bin width in real space of the dimension to be filtered.
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
        area = np.ones(data.shape[-1], dtype=np.int)
        uthresh, lthresh = calc_width(filter_size, real_delta, data.shape[-1])
        area[uthresh:lthresh] = 0

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
            for i in range(data.shape[0]):
                if _w[i, 0] < skip_wgt:
                    _d_cl[i] = 0  # skip highly flagged (slow) integrations
                    _d_res[i] = _d[i]
                    info.append({'skipped': True})
                else:
                    _cl, info_here = aipy.deconv.clean(_d[i], _w[i], area=area, tol=tol, stop_if_div=False, maxiter=maxiter, gain=gain)
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
        a1 = np.ones(data.shape[0], dtype=np.int)
        uthresh, lthresh = calc_width(filter_size[0], real_delta[0], data.shape[0])
        a1[uthresh:lthresh] = 0
        a2 = np.ones(data.shape[1], dtype=np.int)
        uthresh, lthresh = calc_width(filter_size[1], real_delta[1], data.shape[1])
        a2[uthresh:lthresh] = 0
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
    lst_select = np.logical_and(np.unique(data.lst_array) >=  lst_min * 2. * np.pi / 24.,
                                np.unique(data.lst_array) <= lst_max * 2. * np.pi / 24.)
    times_select = np.unique(data.time_array)[lst_select]
    ntimes = len(times_select)
    if np.mod(ntimes,2)==1 and ntimes>1:
        ntimes -=1
        times_select = times_select[:-1]

    #make sure the number of time-steps is even
    ntimes = len(np.unique(data.time_array))
    if np.mod(ntimes,2)==1 and ntimes>1:
        ntimes -=1
        times_select = np.unique(data.time_array)[:-1]
    #Select frequencies between fmin and fmax
    freqs = data.freq_array.squeeze()
    select_channels = np.where(np.logical_and(freqs>=fmin,freqs<=fmax))[0]
    if np.mod(len(freqs[select_channels]),2)==1:
        select_channels=select_channels[:-1]
    data = data.select(freq_chans=select_channels,times = times_select,inplace=False)
    return data

def get_corr_data(data,corrkey):
    '''
    retrieve data and flags from uv data for a single polarization
    and antenna pair.
    Args:
        data, pyuvdata object
        corrkey, 3-tuple (ant0, ant1, pol)
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


    darray = data.data_array[selection,:,:,pol].squeeze()
    wghts = data.flag_array[selection,:,:,pol].squeeze()

    return wghts, darray

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



def filter_data_clean(data,data_d,corrkey,fmin = 45e6, fmax = 85e6,
                     fringe_rate_max = .2e-3, delay_max = 600e-9,
                     lst_min = None,lst_max=None,taper='boxcar',filt2d_mode='rect',
                     add_clean_components=True,freq_domain = 'delay',
                     time_domain = 'time',tol=1e-7,bad_wghts=False,
                     flag_across_time = True,bad_resid=False,
                     fringe_rate_filter = False,acr=False):
    '''
    data, pyuvdata object storing summed measurement
    data_d, pyuvdata object storing differential measurement
    corrkey, tuple for correlation (ant1,ant2,pol)
    fmin, minimum frequency (Hz), float
    fmax, maximum frequency (Hz), float
    fringe_rate_max, maximum fringe_rate to clean (Hz), float, default = .2e-3 sec
    delay_max, maximum delay to clean (sec), float, default = 600e-9 sec
    lst_min, minimum lst to run waterfall from -- !!BREAKS IF DATA CROSSES 0 LST!!
    lst_max, maximum lst to run waterfall from -- !!BREAKS IF DATA CROSSES 0 LST!!
    taper, string, Fourier window function.
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
    '''
    data = down_select_data(data,fmin,fmax,lst_min,lst_max)
    data_d = down_select_data(data_d,fmin,fmax,lst_min,lst_max)

    freqs = data.freq_array.squeeze()
    delays = fft.fftshift(fft.fftfreq(len(freqs),freqs[1]-freqs[0]))

    times = 3600.*24.*np.unique(data.time_array)
    fringe_rates = fft.fftshift(fft.fftfreq(len(times),times[1]-times[0]))

    wghts,darray = get_corr_data(data, corrkey)
    _, darray_d = get_corr_data(data_d, corrkey)

    if flag_across_time:
        for fnum in range(len(freqs)):
            wghts[:,fnum] = np.any(wghts[:,fnum])
    wghts = np.invert(wghts).astype(float)
    if fringe_rate_filter:
        delta_bin = [times[1]-times[0],freqs[1]-freqs[0]]
        filter_size = [fringe_rate_max,delay_max]
        model,resid,info = high_pass_fourier_filter(darray,wghts,filter_size,
                                                          delta_bin,tol=tol,add_clean_residual=acr,
                                                          clean2d=True,window=taper,filt2d_mode=filt2d_mode,
                                                          bad_resid=bad_resid,bad_wghts=bad_wghts)
        model_d,resid_d,info_d = high_pass_fourier_filter(darray_d,wghts,filter_size,
                                                          delta_bin,tol=tol,add_clean_residual=acr,
                                                          clean2d=True,window=taper,bad_wghts=bad_wghts,
                                                          filt2d_mode=filt2d_mode,bad_resid=bad_resid)
    else:
        delta_bin = freqs[1]-freqs[0]
        filter_size = delay_max

        model,resid,info = high_pass_fourier_filter(darray,wghts,filter_size,
                                                          delta_bin,add_clean_residual=acr,
                                                          bad_wghts = bad_wghts,
                                                          tol=tol,window=taper,bad_resid=bad_resid)

        model_d,resid_d,info_d = high_pass_fourier_filter(darray_d,wghts,filter_size,
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

    else:
        x = freqs
    xg,yg = np.meshgrid(x,y)
    return xg,yg,output,output_d


WMAT_CACHE = {}
def clear_cache():
    WMAT_CACHE = {}

def linear_filter(freqs,ydata,flags,patch_c = [], patch_w = [], filter_factor = 1e-3,weights='I',
                  renormalize=True,zero_flags=True,taper='boxcar',cache = WMAT_CACHE):
    '''
    a linear delay filter that suppresses modes within the wedge by a factor of filter_factor.
    freqs, nchan vector of frequencies
    ydata, nchan vector of complex data
    flags, nchan bool vector of flags
    '''
    if filter_factor == 0:
        weights = 'I'
    if isinstance(patch_c, float):
        patch_c = [patch_c]
    if isinstance(patch_w, float):
        patch_w = [patch_w]
    nf = len(freqs)
    #print(nf)
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

            if zero_flags:
                cmat[:,flags]=0.
                cmat[flags,:]=0.
            #print(np.linalg.cond(cmat))
            wmat = np.linalg.pinv(cmat)*filter_factor
            cache[wkey]=wmat
        else:
            wmat = cache[wkey]
    output = ydata
    #output = fft.fft(np.dot(wmat,output * taper))
    output = np.dot(wmat,output)
    #output = fft.ifft(output * taper)
    #output = fft.fft(output)/taper

    return output


def filter_data_linear(data,data_d,corrkey,fmin = 45e6, fmax = 85e6,
                     fringe_rate_max = .2e-3, delay_max = 600e-9, delay_center = 0.,
                     lst_min = None,lst_max=None,taper='boxcar',
                     freq_domain = 'delay', zero_flags = True,
                     time_domain = 'time',tol=1e-7,
                     flag_across_time = True,
                     fringe_rate_filter = False, cache = WMAT_CACHE):
    '''
    data, pyuvdata object storing summed measurement
    data_d, pyuvdata object storing differential measurement
    corrkey, tuple for correlation (ant1,ant2,pol)
    fmin, minimum frequency (Hz), float
    fmax, maximum frequency (Hz), float
    fringe_rate_max, maximum fringe_rate to clean (Hz), float, default = .2e-3 sec
    delay_max, maximum delay to clean (sec), float, default = 600e-9 sec
    lst_min, minimum lst to run waterfall from -- !!BREAKS IF DATA CROSSES 0 LST!!
    lst_max, maximum lst to run waterfall from -- !!BREAKS IF DATA CROSSES 0 LST!!
    taper, string, Fourier window function.
    freq_domain, specify if the output should be in the "frequency" or "delay" domain
                string
    time_domain, specify if the output should be in the "time" or "fringe-rate" domain
                string
    flag_across_time, if True, flags in frequency are the union of all flags
                      across time at that frequency
    fringe_rate_filter, if True, clean in 2d and filter out fringe-rate modes.
                        if False, only clean per time in delay space.
    '''

    data = down_select_data(data,fmin,fmax,lst_min,lst_max)
    data_d = down_select_data(data_d,fmin,fmax,lst_min,lst_max)

    freqs = data.freq_array.squeeze()
    delays = fft.fftshift(fft.fftfreq(len(freqs),freqs[1]-freqs[0]))

    times = 3600.*24.*np.unique(data.time_array)
    fringe_rates = fft.fftshift(fft.fftfreq(len(times),times[1]-times[0]))
    ntimes = len(times)
    nfreq = len(freqs)

    wghts,darray = get_corr_data(data, corrkey)
    _, darray_d = get_corr_data(data_d, corrkey)

    if flag_across_time:
        for fnum in range(len(freqs)):
            wghts[:,fnum] = np.any(wghts[:,fnum])

    if not isinstance(delay_max,list):
        delay_widths = [delay_max]
    if not isinstance(delay_center,list):
        delay_centers = [delay_center]
    resid = np.zeros_like(darray)
    resid_d = np.zeros_like(darray_d)

    for tnum in range(ntimes):
        resid[tnum,:] = linear_filter(freqs,darray[tnum,:],wghts[tnum,:],patch_c = delay_center,
                                     patch_w = delay_max, filter_factor = tol, weights = 'WTL',
                                     renormalize = False, zero_flags = zero_flags,
                                     taper = taper)

        resid_d[tnum,:] = linear_filter(freqs,darray_d[tnum,:],wghts[tnum,:],patch_c = delay_center,
                                     patch_w = delay_max, filter_factor = tol, weights = 'WTL',
                                     renormalize = False, zero_flags = zero_flags,
                                     taper = taper)


    if fringe_rate_filter:
        for cnum in range(nfreq):
            resid[:,cnum] = linear_filter(times,resid[:,cnum],wghts[:,cnum],patch_c = [0.],
                                     patch_w = [max_fringe_rate], filter_factor = tol, weights = 'WTL',
                                     renormalize = False, zero_flags = zero_flags,
                                     taper = taper)
            resid_d[:,cnum] = linear_filter(times,resid_d[:,cnum],wghts[:,cnum],patch_c = [0.],
                             patch_w = [max_fringe_rate], filter_factor = tol, weights = 'WTL',
                             renormalize = False, zero_flags = zero_flags,
                             taper = taper)
    output = resid
    output_d = resid_d

    if time_domain == 'fringe-rate':
        y = fringe_rates
        taper = signal.windows.get_window(taper, ntimes)
        taper /= np.sqrt(np.mean(taper**2.))
        taper = np.array([taper for m in range(nfreq)]).T
        output = fft.fftshift(fft.ifft(fft.fftshift(taper * output,axes=[0]),axis=0),axes=[0])
        output_d = fft.fftshift(fft.ifft(fft.fftshift(taper * output_d,axes=[0]),axis=0),axes=[0])
    else:
        y = np.unique(data.lst_array) * 24. / 2. / np.pi
    if freq_domain == 'delay':
        x = delays
        taper = signal.windows.get_window(taper, nfreq)
        taper = np.array([taper for m in range(ntimes)])
        taper /= np.sqrt(np.mean(taper**2.))
        output = fft.fftshift(fft.ifft(fft.fftshift(output * taper,axes=[1]),axis=1),axes=[1])
        output_d = fft.fftshift(fft.ifft(fft.fftshift(output_d * taper,axes=[1]),axis=1),axes=[1])
    elif freq_domain =='frequency':
        x = freqs
    else:
        raise ValueError("Invalid output domain provided. Must be 'frequency' or 'delay'")
    xg,yg = np.meshgrid(x,y)
    return xg,yg,output,output_d


def integrate_LST(data, data_d, corrkey,  fmin = 45e6, fmax=85e6, fringe_rate_max = .2e-3,
              delay_max = 300e-9, delay_center = 0., lst_min = None, lst_max = None, taper = 'boxcar',
              freq_domain = 'delay', zero_flags = True,
              tol = 1e-7, flag_across_time = True, fringe_rate_filter = False, filter_method = 'linear',
              add_clean_components = True, avg_coherent = True, sq_units = True, cache = WMAT_CACHE):
    '''
    integrate data over LST from a single baseline.
    Args:
        data, pyuvdata object representing data
        data_d, pyuvdata object storing diffed data
        corrkey, key selecting baseline (ant0, ant1, pol)
        fmin, minimum frequency
        fmax, maximum frequency
        fringe_rate_max, maximum fringe rate to filter below
        filter_method, string set to 'linear' or 'clean' and determines the
                       method to clean at.
        add_clean_components, if True, add clean components to data_array
                             does nothing if filtering is 'linear'.
        avg_coherent, boolean, if True integrate coherently.
        sq_units, if True, use square units (product of even/odd data FT).
        cache, dictionary containing filtering matrices.
    '''
    if filter_method == 'linear':
        xg, yg, darray, darray_d = filter_data_linear(data = data ,data_d = data_d,corrkey = corrkey, fmin = fmin, fmax = fmax,
                             fringe_rate_max = fringe_rate_max, delay_max = delay_max, delay_center = delay_center,
                             lst_min = lst_min, lst_max=lst_max, taper=taper,
                             freq_domain = freq_domain, zero_flags = zero_flags,
                             time_domain = "time",tol=tol,
                             flag_across_time = flag_across_time,
                             fringe_rate_filter = fringe_rate_filter)

    elif filter_method == 'clean':
        xg, yg, darray, darray_d = filter_data_clean(data = data,data_d = data_d,corrkey = corrkey,fmin = fmin, fmax = fmax,
                             fringe_rate_max = fringe_rate_max, delay_max = delay_max,
                             lst_min = lst_min,lst_max=lst_max,taper=taper,filt2d_mode='rect',
                             add_clean_components=add_clean_components,freq_domain = freq_domain,
                             time_domain = "time",tol=tol,bad_wghts=False,
                             flag_across_time = flag_across_time,bad_resid=False,
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

def filter_and_average_abs(data, data_d, corrkey, fmin=45e6, fmax = 85e6, fringe_rate_max = .2e-3, delay_max = 300e-9, delay_center = 0.,
                           lst_min = None, lst_max = None, taper = 'boxcar', freq_domain = 'delay', zero_flags = True,
                           tol = 1e-7, flag_across_time = True, fringe_rate_filter = False, filter_method = 'linear',
                           add_clean_components = True, avg_coherent = True, sq_units = True, cache = WMAT_CACHE):
    '''
    delay filter data and compute average.
    data, pyuvdata data set
    data_d, pyuvdata diffed data set
    corrkey, 3-tuple or list (ant0, ant1, pol num)
    fmin, minimum frequency (Hz)
    fmax, maximum frequency (Hz)
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
    cache: dictionary storing linear filter matrices. Ignored if filter_method = 'clean'
    '''
    if filter_method == 'linear':
        xg, yg, darray, darray_d = filter_data_linear(data = data ,data_d = data_d,corrkey = corrkey, fmin = fmin, fmax = fmax,
                             fringe_rate_max = fringe_rate_max, delay_max = delay_max, delay_center = delay_center,
                             lst_min = lst_min, lst_max=lst_max, taper=taper,
                             freq_domain = freq_domain, zero_flags = zero_flags,
                             time_domain = "time",tol=tol,
                             flag_across_time = flag_across_time,
                             fringe_rate_filter = fringe_rate_filter)

    elif filter_method == 'clean':
        xg, yg, darray, darray_d = filter_data_clean(data = data,data_d = data_d,corrkey = corrkey,fmin = fmin, fmax = fmax,
                             fringe_rate_max = fringe_rate_max, delay_max = delay_max,
                             lst_min = lst_min,lst_max=lst_max,taper=taper,filt2d_mode='rect',
                             add_clean_components=add_clean_components,freq_domain = freq_domain,
                             time_domain = "time",tol=tol,bad_wghts=False,
                             flag_across_time = flag_across_time,bad_resid=False,
                             fringe_rate_filter = fringe_rate_filter,acr=False)
    else:
        raise ValueError("Failed to specify a valid filtering method. Valid options are 'clean' and 'linear'")
    #split data into even and odd sets.

    darray_even = (darray + darray_d) / 2.
    darray_odd = (darray - darray_d) / 2.
    darray_d_even =  darray_d[::2]
    darray_d_odd = darray_d[1::2]

    if avg_coherent:
        darray_even = np.mean(darray_even, axis = 0)
        darray_odd = np.mean(darray_odd, axis = 0)

        darray_d_even = np.mean(darray_d_even, axis = 0)
        darray_d_odd = np.mean(darray_d_odd, axis = 0)

    darray_d = darray_d_even * np.conj(darray_d_odd ) / 4.
    darray = darray_even * np.conj(darray_odd)

    if not avg_coherent:
        darray = np.mean(darray, axis = 0)
        darray_d = np.mean(darray_d, axis = 0) * np.sqrt(2.)

    if not sq_units:
        darray = sqrt_abs(darray)
        darray_d = sqrt_abs(darray_d)

    return xg[0,:], darray, darray_d


def avg_comparison_plot(plot_dict_list, sq_units = True,freq_domain = 'delay', ylim = [None, None],
                                xlim = [None,None],logscale = True, legend_font_size = 14,
                                label_font_size = 14, tick_font_size = 14):
    '''
    plot_dict_list: a list of dictionaries specifying the plotting parameters of each line.
    each dictionary must have the following:
        DATA, a pyuvdata object containing primary data.
        DATA_DIFF, a pyuvdata object containing diffed data.
        CORRKEY (ant0, ant1, pol)
        LINESTYLE, linestyle to use
        COLOR, color of line
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

    freq_domain, string, specify if output is in "frequency" or "delay" domain.
    ylim, 2-tuple with upper and lower bounds on plot. If bound is None, will be rounded
          to nearest order of magnitude.
    xlim, 2-tuple with upper and lower x-limits on plot.
    sq_units, if True, show the square of the delay-transform. If false, show the
             square root of the absoute value.
    logscale, if True, y-axis is logarithmically scaled.
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
        x, y, yd = filter_and_average_abs(data = pd['DATA'], data_d = pd['DATA_DIFF'],
                                corrkey = pd['CORRKEY'], fmin = pd['FMIN'],
                                fmax = pd['FMAX'], fringe_rate_max = pd['FRINGE_RATE_MAX'],
                                delay_max=pd['DELAY_WIDTHS'], delay_center = pd['DELAY_CENTERS'],
                                lst_min = pd['LST_MIN'], lst_max = pd['LST_MAX'],
                                taper = pd['TAPER'], freq_domain = freq_domain,
                                zero_flags = pd['ZERO_FLAGS'], tol = pd['TOL'],
                                flag_across_time = pd['FLAG_ACROSS_TIME'],
                                fringe_rate_filter = pd['FRINGE_RATE_FILTER'],
                                filter_method = pd['FILTER_METHOD'],
                                add_clean_components = pd['ADD_CLEAN_MODEL'],
                                avg_coherent = pd['AVG_COHERENT'],
                                sq_units = sq_units, cache = pd['CACHE'])

        if freq_domain == 'delay':
            x *= 1e9
        elif freq_domain == 'frequency':
            x *= 1e-6
        lines.append(plt.plot(x,np.abs(y),lw=pd['LINEWIDTH'],
                              color=pd['COLOR'],
                              ls=pd['LINESTYLE'])[0])
        plt.plot(x,np.abs(yd),lw=1,ls=pd['LINESTYLE'],
                 color=pd['COLOR'])
        labels.append(pd['LABEL'])
        if pd['SHOW_HORIZON'] and freq_domain == 'delay':
            hzn = get_horizon(pd['DATA'],(pd['CORRKEY'][0],pd['CORRKEY'][1]))
            plt.axvline(hzn*1e9, ls = pd['LINESTYLE'], color = [.5,.5,.5])
            plt.axvline(-hzn*1e9, ls = pd['LINESTYLE'], color = [.5,.5,.5])

        if pd['SHOW_FILTER'] and freq_domain == 'delay':
            plt.axvline(pd['DELAY_WIDTHS']*1e9, ls = pd['LINESTYLE'], color = [.75,.75,.75])
            plt.axvline(-pd['DELAY_WIDTHS']*1e9, ls = pd['LINESTYLE'], color = [.75, .75, .75])


        if ylim_in[0] is None:
            if np.abs(y).max() > ylim[1]:
                ylim[1] = np.abs(y).max()
        if ylim_in[1] is None:
            if np.abs(yd).mean() < ylim[0]:
                ylim[0] = np.abs(yd).mean()

        ylim[1] = 10.**np.ceil(np.log10(ylim[1])) * 10.
        ylim[0] = 10.**np.floor(np.log10(ylim[0])) / 10.

        if xlim_in[1] is None:
            if x.max() > xlim[1]:
                xlim[1] = x.max()
        if xlim_in[0] is None:
            if x.min() < xlim[0]:
                xlim[0] = x.min()


    #print(x.max())
    #print(x.min())
    #print(xlim)
    #print(ylim)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if logscale:
        plt.yscale('log')

    plt.legend(lines,labels,loc='best',fontsize=16)
    if freq_domain == 'delay':
        if pd['SHOW_K']:
            #plot k-parallel axis above plot if we are in the delay domain.
            f0 = (pd['FMIN'] + pd['FMAX'])/2.
            z0 = 1420.41e6/f0 - -1.
            y0 = 3e5/100. * (1.+z0)**2. / np.sqrt(.7 + .3 * (1.+z0)**3.) / 1420.41e6
            ax1=plt.gca()
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
                #print(tick)
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
