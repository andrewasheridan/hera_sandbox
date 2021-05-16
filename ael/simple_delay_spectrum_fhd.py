
### Do delay spectrum analysis without using hera_pspec

from scipy.io import readsav
from scipy.stats import binned_statistic
import numpy as np
import sys, yaml, glob, os, time, optparse
from eorsky import comoving_voxel_volume 
from pyuvdata import UVData
import pylab as pl
from sandbox_cosmo_funcs import *

### Define function to save and quit.

### Parse command line args

o = optparse.OptionParser()
o.add_option('-r', "--random_vis", action='store_true', help="Replace data with random visibilities of comparable variance.")
o.add_option('-p', "--partition_averaging", dest='part', action='store_true', help="Partition the time series, average partitions, and save variances per k.")
o.add_option('-c', '--covariance', dest='covar', action='store_true', help='Calculate and return the covariance of the data array')
#o.add_option('-f', '--freq_chan', dest='freq', help='Get covariance for a single frequency channel', default=None, type=int)
o.add_option('-f', '--freq_chan', dest='freq', action='store_true', help='Get covariances, then average in frequency.')

opts,args = o.parse_args(sys.argv[1:])

partition = opts.part
over_k = not opts.part

if len(args) == 1:
    config_file = str(args[0])
else:
    print("Command takes one argument: config_file")
    sys.exit(1)

with open(config_file, 'r') as cfile:
    try:
        cfg = yaml.load(cfile)
    except yaml.YAMLError as exc:
        raise(exc)

data_cfg = cfg['data']
pspec_cfg = cfg['pspec']
out_cfg = cfg['out']
try:
    catalog_cfg = cfg['catalog']
except KeyError:
    catalog_cfg = {}

out_cfg.update(opts.__dict__)

#-------------------------------------------------------------------------------
# Get obs (and Opp) from first metadata
#-------------------------------------------------------------------------------
if 'obs' in data_cfg.keys():
    obsfile = data_cfg['obs']
else:
    #Take the first recognized obs file.
    obsfile = glob.glob( os.path.join(data_cfg['root'], 'metadata/*_obs.sav') )[0]
 
obs = readsav(obsfile)['obs']
#Op = obs['PRIMARY_BEAM_AREA'][0]   # Shape = (npol, nfreq), type=float32. This is from FHD.
Opp = obs['PRIMARY_BEAM_SQ_AREA'][0]

n_freq_obs = obs['n_freq'][0]
freq_center_obs = obs['freq_center'][0]
freq_res_obs = obs['freq_res'][0]
off = (n_freq_obs/2.)*freq_res_obs
freq_obs = np.linspace( freq_center_obs-off, freq_center_obs+off, n_freq_obs)
Bandwidth = freq_obs.size * freq_res_obs
print("Obs bandwidth (MHz): ", Bandwidth/1e6)

#-------------------------------------------------------------------------------
# Prepare list of data files
#-------------------------------------------------------------------------------

save_dict={}    # results to save to npz

files=[]
for i in range(len(data_cfg['subdirs'])):
    files += glob.glob( os.path.join(data_cfg['root'], 
                                     data_cfg['subdirs'][i], 
                                     data_cfg['template']) )
print("Found %d files." % len(files))


# Load all miriad datafiles into UVData objects

# Select baselines from parameters:
bl = eval(pspec_cfg['baseline'])
bls = [bl]


#-------------------------------------------------------------------------------
# Load data files into memory
#-------------------------------------------------------------------------------
print("Loading data files...")
t0 = time.time()

#def sortfunc(s):
#    s = s.split('/')[-1]
#    return int(s.split('_')[2])
#
#files.sort(key=sortfunc)

dsets = []
data = None
count = 0
if len(files) == 0:
    raise ValueError("Couldn't find any files")
for f in files:
    print count, f
    _d = UVData()
    _d.read_miriad(f, bls=bls)
#    _d.polarization_array = uv.utils.polnum2str(_d.polarization_array)
    if _d.vis_units == "JY":
        _d.vis_units = 'Jy'
    _d.object_name = 'Zenith'
    if data is None: data = _d
    else: data += _d
    count+=1
#    dsets.append(_d)
print("Loaded data in %1.1f sec." % (time.time() - t0))
print("Phased?: ", data.phase_type)
max_chan = data.Nfreqs
if 'chan_range' in data_cfg:
    cmin,cmax = map(lambda x: int(x.strip()), data_cfg['chan_range'].split("_"))
    chans = np.arange(cmin, cmax)
else:
    chans = np.arange(max_chan)

data.select(freq_chans=chans)
Bandwidth = data.freq_array[0, -1] - data.freq_array[0,0]
print 'Bandwidth (MHz): ' , Bandwidth/1e6


print np.unique(data.baseline_array)

##Normal distribution
if opts.random_vis:
    sigma = np.sqrt(np.var(data.data_array))
    mu = np.mean(data.data_array)
    print 'random_vis: mu={}, sigma={}'.format(mu, sigma)
    data.data_array = (np.random.normal(mu, sigma, size=data.data_array.shape) + (1j)*np.random.normal(mu, sigma, size=data.data_array.shape))/np.sqrt(2.)

# Get k_parallel
Zs = 1420e6/data.freq_array[0] - 1
Zmean = np.mean(Zs)
etas = np.fft.fftfreq(data.Nfreqs, d=data.channel_width)
k_parallel = dk_deta(Zmean) * etas


vis_xx = data.data_array[:,0,:,0]
vis_yy = data.data_array[:,0,:,1]
# Remove flagged visibilities
inds = np.where(vis_xx == 0.0)
vis_xx = np.delete(vis_xx, inds[0], axis=0)
vis_yy = np.delete(vis_yy, inds[0], axis=0)

if 'opp' in data_cfg:
    Opp_I = data_cfg['opp'] * np.ones_like(data.Nfreqs)
Opp_I = Opp[0]  # Shape (Nfreqs)
vis_I = vis_xx + np.conj(vis_yy)

Opp_I = Opp_I[chans]

#Convert to K*str
mK =  'mK' in pspec_cfg
vis_I = vis_I * jy2Tstr(data.freq_array, mK = mK)

# FFT
dnu = np.diff(data.freq_array[0])[0]
_visI = np.fft.ifft(vis_I, axis=1)
print("dnu", dnu)
dspec_instr = _visI*_visI.conj()

# Cosmological Normalization:
#Oeff = Op_I**2/Opp_I
scalar = X2Y(Zmean) * (Bandwidth / Opp_I)
dspec_I =  dspec_instr * scalar
Nbins=500
def diagonal_bin(times_jd, covar_mat, Nbins=Nbins):
    """
        Bin the covariance matrix in time lag.
    """
    times_min = (times_jd - times_jd[0])*24.*60.
    Ntimes = times_min.size
    time_per_sample = np.diff(times_min)[0]

    inds = np.arange(Ntimes)
    xind, yind = np.meshgrid(inds, inds)
    diff_inds = xind-yind
    means_real, bins, binnums = binned_statistic(diff_inds.flatten(), np.real(covar_mat).flatten(), bins=Nbins)
    means_imag, bins, binnums = binned_statistic(diff_inds.flatten(), np.imag(covar_mat).flatten(), bins=Nbins)
    bins = time_per_sample * (bins[1:] + bins[:-1])/2.  #Bin centers, in minutes
    return bins, means_real + (1j)*means_imag

if opts.covar:
    # Calculate the covariance matrix of the data_array
    #if opts.freq is None:
    time_arr = np.delete(data.time_array, inds[0])
    if not  opts.freq:
        ## TODO --- Square vis_I to remove phasing?
        covar = np.corrcoef(vis_I)
        save_dict['covar_data'] = covar
        # Calculate the covariance matrix of the delay spectrum and save to file, then exit.
        covar = np.corrcoef(dspec_I)
        save_dict['covar_dspec'] = covar
    else:
        # Keep frequency axis, bin in lag here.
        #covar = np.outer(vis_I[:,opts.freq], vis_I[:,opts.freq].conj())
#        covar = np.zeros((vis_I.shape[0], vis_I.shape[0], vis_I.shape[1]), dtype = np.complex128)
        Nfreq = vis_I.shape[1]
        time_arr = np.delete(data.time_array, inds[0])
        bins, cs = np.zeros((2, Nfreq, Nbins), dtype=np.complex128)
        for fi in range(vis_I.shape[1]):
            covar = np.outer(vis_I[:,fi], vis_I[:,fi].conj())
            bins[fi], cs[fi] = diagonal_bin(time_arr, covar)
        save_dict['Nfreq'] = Nfreq
#        covar = np.einsum("ij,kl->ikl", vis_I, vis_I.conj())
#        import IPython; IPython.embed()
#        for fi in range(covar.shape[2]):
#        bins = np.average(bins, axis=0)
#        cs   = np.average(cs, axis=0)
        save_dict['covar_data'] = cs
        time_arr = np.average(bins, axis=0)
    version = data_cfg['root'].split('/')[-1]
    save_dict['times'] = time_arr
    save_results(save_dict, data_cfg, out_cfg)

# Get a reference line, if available
if 'Nside' in catalog_cfg:
    if 'sigma' in catalog_cfg:
        nside = catalog_cfg['Nside']
        sigma = catalog_cfg['sigma']
        om = 4*np.pi/(12.*nside**2)
        dnu = np.diff(freq_obs)[0]/1e6   #MHz
        ref_line = comoving_voxel_volume(Zmean, dnu, om) * (sigma)**2
if 'furlanetto_flat' in catalog_cfg:
    f = readsav('/users/alanman/FHD/catalog_data/eor_power_1d.idlsave')
    maxpow = max(f['power'])
    ref_line = maxpow/0.7**3   #Furlanetto's curve includes little-h
if 'npz' in catalog_cfg:
    # Setting reference level to the most averaged spectrum in an npz file.
    f = np.load(catalog_cfg['npz'])
    try:
        ref_avg_mode = f['avg_mode']
    except KeyError:
        ref_avg_mode = 'over_k'

    ref_avgs = f['avgs']
    if ref_avg_mode == 'over_k':
        print ref_avgs.shape
        ref_line = np.average(ref_avgs[-1])
    if ref_avg_mode == 'partitioned':
        print ref_avgs.shape
        ref_line = np.average(ref_avgs[-1, -1])
try:
    save_dict['ref_line'] = ref_line
except NameError:
    pass

## Plot pspecs with different levels of averaging:

#lens = [ i*3600./data.integration_time for i in range(1,25)]
#lens = map(int, lens)
#avg_lens_time = (data.time_array - data.time_array[0]) * (24*60.)  # Minutes
#avg_lens_nsamp = avg_lens_time/(data.integration_time/60.)
#avg_lens_nsamp = np.floor(avg_lens_nsamp).astype(int)
avg_lens_nsamp = np.arange(2, data.Ntimes - len(inds))
avg_lens_time = avg_lens_nsamp * data.integration_time[0]/60.   #Minutes
# Redefine the avg_lens_time to account for missing times
#avg_lens_nsamp = np.delete(avg_lens_nsamp)[2:]
#avg_lens_ntime = np.delete(avg_lens_nsamp, inds[0])[2:]
avgs = []
variances = []
lens = avg_lens_nsamp
nsamps = []

#import IPython; IPython.embed()
for i,l in enumerate(lens):
    if partition:
        parts = np.array([dspec_I[j*l:(j+1)*l,:] for j in range(int(np.floor(data.Ntimes/float(l))))])
        parts = [ p for p in parts if len(p) == l]
        nsamps.append(len(parts))
        avgs.append(np.mean(parts, axis=1))
    elif over_k:
        avgs.append(np.mean(dspec_I[0:l,:], axis=0))
        nsamps.append(dspec_I.shape[1])

#avgs_second = [np.mean(dspec_I[0:l,:], axis=0) for l in xrange(dspec_I.shape[0])]
print("Mean pspec_instr: {:.4e}".format(np.mean(dspec_instr)))
print("Scalar=", np.mean(scalar))
print("Scalar*mean_pspc_instr=", np.mean(scalar)*np.mean(dspec_instr))
print "Avg/Ref: ", np.mean(avgs[-1])/ref_line
print 'Integration time = {}'.format(data.integration_time)
print 'Compare: {} samples vs. {} = length of dspec_I'.format(lens[-1], dspec_I.shape[0])

save_dict.update(dict(k_parallel=k_parallel, avgs=avgs, avg_lens_nsamp=lens, avg_lens_time=avg_lens_time, dspec_instr=dspec_instr, scalar=scalar, bl=bls, nsamps=nsamps))

save_results(save_dict, data_cfg, out_cfg)

