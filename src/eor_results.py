import numpy as n,os
import cosmo_units
from cosmo_units import f212z, c
import glob
import ipdb
import matplotlib.pyplot as p
import numpy as np
import itertools
# from twentyonecmfast_tools import load_andre_models, all_and

#measurements

def errorbars(data,axis=1,per=95):
    mean = n.percentile(data, 50, axis=axis)
    lower = mean - n.percentile(data, 50-per/2., axis=axis)
    upper = n.percentile(data, 50+per/2., axis=axis) - mean
    return lower, upper

def MWA_128_beardsley_2016_all(pol='EW'):
    '''
    Results from MWA Beardsley 2016. ~60hours

    outputs results[z] = n.array([k,Delta^2,2-sigma upper, 2-sigma lower])
    '''
    from astropy.table import Table
    DATA = Table.read(os.path.dirname(__file__)+'/data/MWA_128T_Beardsley_2016.txt',format='ascii')
    results = {}
    for rec in DATA:
        if rec['pol']!=pol:continue
        try:
            results[rec['redshift']].append([rec['k'],rec['Delta2'],rec['Delta2_err'],0])
        except(KeyError):
            results[rec['redshift']] = [[rec['k'],rec['Delta2'],rec['Delta2_err'],0]]
    for z in results:
        results[z] = n.array(results[z])
    return results

def load_andre_models():
    """Get Arrays of parms, ks, delta^2 and err from 21cmfast output.

    Input a string that globs to the list of input model files
    return arrays of parameters,k modes, delta2,and delt2 error
    parm_array expected to be nmodels,nparms
    with columns (z,Nf,Nx,alphaX,Mmin,other-stuff....)
    delta2_array expected to be nmodels,nkmodes
    """
    filenames = glob.glob(os.path.dirname(__file__)+'/data/21cmfast/ps*')
    filenames.sort()
    parm_array = []
    k_array = []
    delta2_array = []
    delta2_err_array = []
    for filename in filenames:
        parms = os.path.basename(filename).split('_')
        if parms[0].startswith('reion'):
            continue
        parm_array.append(map(float, [parms[3][1:],
                                      parms[4][2:],  # Nf
                                      parms[6][2:],  # Nx
                                      parms[7][-3:],  # alphaX
                                      parms[8][5:],  # Mmin
                                      parms[9][5:]]))
        D = np.loadtxt(filename)
        k_array.append(D[:, 0])
        delta2_array.append(D[:, 1])
        delta2_err_array.append(D[:, 2])
    parm_array = np.array(parm_array)
    raw_parm_array = parm_array.copy()
    k_array = np.ma.array(k_array)
    raw_k_array = k_array.copy()
    delta2_array = np.ma.masked_invalid(delta2_array)
    raw_delta2_array = delta2_array.copy()
    delta2_err_array = np.ma.array(delta2_err_array)
    return parm_array, k_array, delta2_array, delta2_err_array

def all_and(arrays):
    """Input list or array, return arrays added together.
    """
    if len(arrays) == 1:
        return arrays
    out = arrays[0]
    for arr in arrays[1:]:
        out = np.logical_and(out, arr)
    return out

def PAPER_32_all():
    '''
    Results from  PAPER 32  Jacobs et.al 2014

    outputs results[z] = n.array([k, Delta^2, 2-sigma upper, 2-sigma lower])
    '''

    PAPER_RESULTS_FILES = glob.glob(os.path.dirname(__file__)+'/data/psa32_apj/pspec_*.npz')
    PAPER_RESULTS_FILES.sort()
    PAPER_RESULTS_FILES = [f for f in PAPER_RESULTS_FILES if '110_149' not in f]
    freqs= []
    for filename in PAPER_RESULTS_FILES:
        try:
            freqs.append(n.load(filename)['freq']*1e3)
        except(KeyError):
            try:
                dchan = int(filename.split('/')[-1].split('.')[0].split('_')[2])-int(filename.split('/')[-1].split('.')[0].split('_')[1])
                chan = int(filename.split('/')[-1].split('.')[0].split('_')[1]) + dchan/2.
                freqs.append(chan/2. + 100) #a pretty good apprximation of chan 2 freq for 500kHz channels
            except: continue
    freqs = n.array(freqs)
    zs = cosmo_units.f212z(freqs*1e6)
    results = {}
    for files,z in zip(PAPER_RESULTS_FILES,zs):
        f=n.load(files)
        results[z] = n.array([f['k'],f['k3pk'],f['k3pk']+f['k3err'],f['k3pk']-f['k3err']]).T
    return results

def PAPER_32_parsons():
    '''
    Results from  PAPER 32  Parsons et.al 2014

    outputs results[z] = n.array([k, Delta^2, 2-sigma upper, 2-sigma lower])
    '''

    PAPER_RESULTS_FILES = glob.glob(os.path.dirname(__file__)+'/data/psa32_apj/pspec_110_149.npz')
    PAPER_RESULTS_FILES.sort()
    freqs= []
    for filename in PAPER_RESULTS_FILES:
        try:
            freqs.append(n.load(filename)['freq']*1e3)
        except(KeyError):
            try:
                dchan = int(filename.split('/')[-1].split('.')[0].split('_')[2])-int(filename.split('/')[-1].split('.')[0].split('_')[1])
                chan = int(filename.split('/')[-1].split('.')[0].split('_')[1]) + dchan/2.
                freqs.append(chan/2. + 100) #a pretty good apprximation of chan 2 freq for 500kHz channels
            except: continue
    freqs = n.array(freqs)
    zs = cosmo_units.f212z(freqs*1e6)
    results = {}
    for files,z in zip(PAPER_RESULTS_FILES,zs):
        f=n.load(files)
        results[z] = n.array([f['k'],f['k3pk'],f['k3pk']+f['k3err'],f['k3pk']-f['k3err']]).T
    return results

def PAPER_64_all():
    '''
    Results from  PAPER 64  Ali et.al 2015

    outputs results[z] = n.array([k, Delta^2, 2-sigma upper, 2-sigma lower])
    '''

    PAPER_RESULTS_FILES = glob.glob(os.path.dirname(__file__)+'/data/psa64_apj/pspec_*.npz')
    PAPER_RESULTS_FILES.sort()
    freqs = []
    for filename in PAPER_RESULTS_FILES:
        try:
            freqs.append(n.load(filename)['freq']*1e3)
        except(KeyError):
            try:
                dchan = int(filename.split('/')[-1].split('.')[0].split('_')[2])-int(filename.split('.')[0].split('_')[1])
                chan = int(filename.split('/')[-1].split('.')[0].split('_')[1]) + dchan/2.
                freqs.append(chan/2. + 100) #a pretty good apprximation of chan 2 freq for 500kHz channels
            except: continue
    freqs = n.array(freqs)
    zs = cosmo_units.f212z(freqs*1e6)
    results = {}
    for files,z in zip(PAPER_RESULTS_FILES,zs):
        f=n.load(files)
        results[z] = n.array([f['k'],f['k3pk'],f['k3pk']+f['k3err'],f['k3pk']-f['k3err']]).T
    return results



def MWA_128_all():
    '''
    MWA_128 data from dillion 2015
    return format will be dict[z] = n.array([[k,Delta^2,top,bottom]]) all in mK^2
    '''
    MWA_RESULTS_FILE=glob.glob(os.path.dirname(__file__)+'/data/mwa128/*.dat')
    zs= []
    results = {}
    for files in MWA_RESULTS_FILE:
        name = files.split('/')[-1]
        nums = name.split('=')[1].split('.')[:2]
        z= float(nums[0]+'.'+nums[1])

        data = n.genfromtxt(files, delimiter=' ')
        results[z] = data[:,[0,1,5,4]]

    return results

def LOFAR_Patil_2017():
    """
    Lofar limits from Patil et al 2017
    """
    LOFAR_Patil = {}
    LOFAR_Patil[8.3] = np.array([[0.053,0,131.5**2,0],
                                [0.067,0,242.1**2,0],
                                [0.083,0,220.9**2,0],
                                [0.103,0,337.4**2,0],
                                [0.128,0,407.7**2,0]])
    LOFAR_Patil[9.15] = np.array([[0.053,0,86.4**2,0],
                                [0.067,0,144.2**2,0],
                                [0.083,0,184.7**2,0],
                                [0.103,0,296.1**2,0],
                                [0.128,0,342.0**2,0]])
    LOFAR_Patil[10.1] = np.array([[0.053,0,79.6**2,0],
                                [0.067,0,108.8**2,0],
                                [0.083,0,148.6**2,0],
                                [0.103,0,224**2,0],
                                [0.128,0,366.1**2,0]])
    return LOFAR_Patil

def MWA_128_beards():
    """MWA_128 data from Beardsley 2016.

    """
    MWA_beards = {}
    MWA_beards[7.1] = n.array([[0.27, 0, 2.7e4, 0]])
    MWA_beards[6.8] = n.array([[0.24, 0, 3.02e4, 0]])
    MWA_beards[6.5] = n.array([[0.24, 0, 3.22e4, 0]])
    return MWA_beards

def z_slice(redshift,pspec_data):
    """
    input a power spectrum data dict output of MWA_32T_all() or GMRT_2014_all()
    returns a slice along k for the input redshift
    example
    z,pspec[k,k3pK] = z_slice(MWA_32T_all())
    """
    zs = n.array(pspec_data.keys())
    closest_z = zs[n.abs(zs-redshift).argmin()]
    return closest_z, pspec_data[closest_z]

def min_limit(pspec_data,krange=None):
    #for each redshift, find the min UL,
    #return a pspec_data
    outdata = {}
    for z in pspec_data:
        if krange is None:
            min_pspec_index = pspec_data[z][:,2].argmin()
        else:
            k = pspec_data[z][:,0]
            p = pspec_data[z][:,2] #the upper limits
            p = np.ma.masked_array(p,mask=[False for x in p], shrink=False)
            pspec_in_k_range = np.ma.masked_where(np.logical_or(k<krange[0],k>krange[1]),p)
            if len(pspec_in_k_range) ==1:
                pspec_in_k_range.mask = [pspec_in_k_range.mask]
            try:
                if all(pspec_in_k_range.mask):
                    if all(k < krange[0]):
                        min_pspec_index = len(k) - 1
                    elif all(k > krange[1]):
                        min_pspec_index = 0
                else:
                    min_pspec_index = pspec_in_k_range.argmin()
            except:
                from IPython import embed
                embed()
        outdata[z] = pspec_data[z][min_pspec_index]
    return outdata

def k_slice(k,pspec_data):
    """
    input a power spectrum data dict output of MWA_32T_all() or GMRT_2014_all()
    returns a slice along z for the input redshift
    example
    zs,pspec[k,k3pK] = k_slice(k,MWA_32T_all())
    """

    zs = n.array(pspec_data.keys())

    k_is = [n.abs(pspec_data[redshift][:,0]-k).argmin() for redshift in zs]
    ks = [pspec_data[redshift][k_i,0] for k_i in k_is]
    power = n.vstack([pspec_data[redshift][k_i,:] for k_i in k_is])
    return zs,power

def MWA_32T_all():
    """
    From Josh Dillon April 10, 2014
    Delta^2  |  2sig bottom bar  |  2sig top bar  |  k  |  z | "Detection"

    or

    2sig top bar   |  [ignore this entry]  |  2sig top bar  |  k  |  z | "2sigUL"

    in the units you requested.

    (my request The ideal format would be a text file listing three things:
    Delta^2 [mk^2], k [hMpc^-1], z [z])


    TREATMENT:
    I will store the delta^2 +/- values.  If he doesn't give me a delta^2 or lower limit, I will put
    Delta^2 = 0 and bottom = -top

    return format will be dict[z] = n.array([[k,Delta^2,top,bottom]]) all in mK^2

    """


    MWA_RESULTS_FILE=os.path.dirname(__file__)+'/data/MWA_32T_Results_Summary.txt'
    lines = open(MWA_RESULTS_FILE).readlines()
    result = {}
    for line in lines:
        if line.endswith('2sigUL'):#if its just an upper limit
            l = map(float,line.split()[:-1])
            z = l[-1]
            try:
                result[z].append([l[3],0,l[0],-l[0]])
            except(KeyError):
                result[z] = [[l[3],0,l[0],-l[0]]]
        else:#if its a "detection" which just means its not consistent with 0, not really EoR but probably foregrounds
            #or systematics
            l = map(float,line.split()[:-1])
            z = l[-1]
            try:
                result[z].append([l[3],l[0],l[0],-l[0]])
            except(KeyError):
                result[z] = [[l[3],l[0],l[2],l[1]]]
    for z in result.keys():
        result[z] = n.array(result[z])
    return result

def MWA_32T_at_z(redshift):
    """
    Input a redshift
    Return the Delta^2(k) power spectrum at the redshift nearest the input redshift
    returns: z,array([k,Delta^2,2sig_upper_limit,2sig_lower_limit])
    """
    data = MWA_32T_all()
    zs = n.array(data.keys())
    closest_z = zs[n.abs(zs-redshift).argmin()]
    return closest_z, data[closest_z]

def MWA_32T_at_k(k):
    """
    Input a k
    Returns the Delta^2(k) power spectrum vs redshift at that k. (all pspec in mk^2)
    return format: z,array([k,Delta^2,2sig_upper_limit,2sig_lower_limit])
    """
    data = MWA_32T_all()
    zs = n.array(data.keys())
    k_is = [n.abs(data[redshift][:,0]-k).argmin() for redshift in zs]
    ks = [data[redshift][k_i,0] for k_i in k_is]
    power = n.vstack([data[redshift][k_i,:] for k_i in k_is])
    return zs,power

def GMRT_2014_all():
    #format {z:[k,Delta^2,2sigUL,2sigLL]}
    return {8.6:n.array(
            [[0.1, 0,2e5,  -2e5],
            [0.13, 0,4e5,  -4e5],
            [0.16, 0,1e5,  -1e5],
            [0.19, 0,1.9e5,-1.9e5],
            [0.275,0,2e5,  -2e5],
            [0.31, 0,4e5,  -4e5],
            [0.4,  0,6e5,  -6e5],
            [0.5,  0,8e4,  -8e4]])}

def get_pk_from_npz(files=None, verbose=False):
    """
    Load output from plot_pk_k3pk.npz and returns P(k)  spectrum.

    Return lists of k, Pk, Pk_err, Delta^2 ordered by decreasing redshift
    Return format: z, k_parallel, Pk, Pk_err
    """
    if files is None:
        print 'No Files gives for loading'
        return [], [], [], []

    if len(n.shape(files)) == 0:
        files = [files]

    freqs = []
    if verbose:
        print "parsing npz file frequencies"
    for filename in files:
        if verbose:
            print filename,
        try:
            if verbose:
                print "npz..",
            freqs.append(n.load(filename)['freq']*1e3)  # load freq in MHz
            if verbose:
                print "[Success]"
        except(KeyError):
            if verbose:
                print "[FAIL]"
            try:
                if verbose:
                    print "looking for path like RUNNAME/chan_chan/I/pspec.npz"
                dchan = (int(filename.split('/')[1].split('_')[1]) -
                         int(filename.split('/')[1].split('_')[0]))
                chan = int(filename.split('/')[1].split('_')[0]) + dchan/2
                freqs.append(chan/2. + 100)
                # a pretty good apprximation of chan 2 freq for 500kHz channels
            except(IndexError):
                if verbose:
                    print "[FAIL] no freq found. Skipping..."

    if len(freqs) == 0:  # check if any files were loaded correctly
        print 'No parsable frequencies found'
        print 'Exiting'
        return [], [], [], []

    if verbose:
        print "sorting input files by frequency"
    files = n.array(files)
    files = files[n.argsort(freqs)]
    freqs = n.sort(freqs)
    if verbose:
        print "found freqs"
    freqs = n.array(freqs)
    if verbose:
        print freqs

    z = f212z(freqs*1e6)
    if verbose:
        print "processing redshifts:", z

    kpars = []
    Pks = []
    Pkerr = []
    for i, FILE in enumerate(files):
        F = n.load(FILE)
        if verbose:
            print FILE.split('/')[-1], z[i]
        Pks.append(F['pk'])
        kpars.append(F['kpl'])
        Pkerr.append(F['err'])
    return z, kpars, Pks, Pkerr

def load_pspecs(files,verbose=False):
    """
        Load output from plot_pk_k3pk.npz and returns Delta^2 spectrum.

        Return lists of k, Delta^2, Delta^2_err ordered by  decreasing redshift
        Return standard dict format {z:np.array([k,delta2,ulim,llim])}
        files : list of pspec npz files
    """

    freqs = []
    if verbose:
        print "parsing npz file frequencies"
    for filename in files:
        if verbose:
            print filename,
        try:
            if verbose:
                print "npz..",
            freqs.append(n.load(filename)['freq']*1e3)  # load freq in MHz
            if verbose:
                print "[Success]"
        except(KeyError):
            if verbose:
                print "[FAIL]"
            try:
                if verbose:
                    print "looking for path like RUNNAME/chan_chan/I/pspec.npz"
                dchan = (int(filename.split('/')[1].split('_')[1]) -
                         int(filename.split('/')[1].split('_')[0]))
                chan = int(filename.split('/')[1].split('_')[0]) + dchan/2
                freqs.append(chan/2. + 100)
                # a pretty good apprximation of chan 2 freq for 500kHz channels
            except(IndexError):
                if verbose:
                    print "[FAIL] no freq found."
                    print " You can't do science without freqs. Exiting"
                    return None

    if verbose:
        print "sorting input files by frequency"
    files = n.array(files)
    files = files[n.argsort(freqs)]
    freqs = np.array(n.sort(freqs))

    if verbose:
        print "found freqs"
        print freqs

    redshifts = f212z(freqs*1e6)
    if verbose:
        print "processing redshifts:", redshifts
    data = {}
    for i, FILE in enumerate(files):
        F = n.load(FILE)
        if verbose:
            print FILE.split('/')[-1], redshifts[i]
        k = F['k']
        Delta2 = k**3/(2*np.pi**2) * F['pIv_fold']
        #Delta2 = np.zeros_like(Delta2)
        Delta2_err = 2*k**3/(2*np.pi**2) *F['pIv_fold_err']
        data[redshifts[i]] = np.array([k,Delta2,
                                    Delta2+Delta2_err,
                                    Delta2-Delta2_err]).T
    return data


def get_k3pk_from_npz(files=None, verbose=False):
    """
    Load output from plot_pk_k3pk.npz and returns Delta^2 spectrum.

    Return lists of k, Delta^2, Delta^2_err ordered by  decreasing redshift
    Return format: z, k_magnitude, Delta^2, Delta^2_err
    """
    if files is None:  # check that files are passed
        print 'No Files given for loading'
        return [], [], [], []

    if len(n.shape(files)) == 0:
        files = [files]

    freqs = []
    if verbose:
        print "parsing npz file frequencies"
    for filename in files:
        if verbose:
            print filename,
        try:
            if verbose:
                print "npz..",
            freqs.append(n.load(filename)['freq']*1e3)  # load freq in MHz
            if verbose:
                print "[Success]"
        except(KeyError):
            if verbose:
                print "[FAIL]"
            try:
                if verbose:
                    print "looking for path like RUNNAME/chan_chan/I/pspec.npz"
                dchan = (int(filename.split('/')[1].split('_')[1]) -
                         int(filename.split('/')[1].split('_')[0]))
                chan = int(filename.split('/')[1].split('_')[0]) + dchan/2
                freqs.append(chan/2. + 100)
                # a pretty good apprximation of chan 2 freq for 500kHz channels
            except(IndexError):
                if verbose:
                    print "[FAIL] no freq found. Skipping..."

    if len(freqs) == 0:  # check if any files were loaded correctly
        print 'No parsable frequencies found'
        print 'Exiting'
        return [], [], [], []

    if verbose:
        print "sorting input files by frequency"
    files = n.array(files)
    files = files[n.argsort(freqs)]
    freqs = n.sort(freqs)

    if verbose:
        print "found freqs"
    freqs = n.array(freqs)
    if verbose:
        print freqs

    z = f212z(freqs*1e6)
    if verbose:
        print "processing redshifts:", z
    k3Pk = []
    k3err = []
    kmags = []
    for i, FILE in enumerate(files):
        F = n.load(FILE)
        if verbose:
            print FILE.split('/')[-1], z[i]
        k3Pk.append(F['k3pk'])
        k3err.append(F['k3err'])
        kmags.append(F['k'])
    return z, kmags, k3Pk, k3err

formats = {'GMRT_2014_all':
            dict(fmt='p', ecolor='C5',
                color='C5', uplims=True,
                label='Paciga, 2013'),
            'MWA_32T_all':
            dict(fmt='r*', uplims=True,
                label='Dillon, 2014'),
            'MWA_128_all':
            dict(fmt='y*', uplims=True,
                label='Dillon, 2015'),
            'MWA_128_beardsley_2016_all':
            dict(fmt='g*', uplims=True,
                label='Beardsley, 2016'),
            'PAPER_32_all':
            dict(fmt='d',color='purple', uplims=True,
                label='Jacobs, 2015'),
            'PAPER_32_parsons':
            dict(fmt='d',color='cyan', uplims=True,
                label='Parsons, 2014'),
            'PAPER_64_all':
            dict(fmt='d',color='grey',
                 uplims=True,
                 label='Ali, 2015'),
            'LOFAR_Patil_2017':
            dict(fmt='bh',uplims=True,
                label='Patil, 2017')
            dict(fmt='0.5.',uplims=True,
                label='Parsons 2015')
            }
krange_table = {'GMRT_2014_all': dict(krange=[.1,.5]),
            'MWA_32T_all': dict(krange=None),
            'MWA_128_all': dict(krange=None),
            'MWA_128_beardsley_2016_all':
            dict(krange=None),
            'PAPER_32_all':
            dict(krange=[.1,.6]),
            'PAPER_32_parsons':
            dict(krange=[.1,.6]),
            'PAPER_64_all':
            dict(krange=[.1,.6]),
            'LOFAR_Patil_2017':
            dict(krange=None)
            }
def plot_lowest_limits(files=None,title='',published = None,
            krange=None,models=True,verbose=False,capsize=3.5,figsize=(10,5),**kwargs):
    """
    Plot all published results, taking the _lowest_ limit from each paper.
    files: set of unpublished results
    title: legend name for input Files
    published: list of functions, default is all known to this function
    k_range: range over which to find min (tuple in h cMpc^-1)
    models: boolean, plot fiducal 21cmfast models (included in capo)
    capsize: defines the size of the upper limits caps.
    verbose: print info while plotting
    """
    if krange:
        print krange
    if published is None:
        published = [MWA_32T_all,MWA_128_all,MWA_128_beardsley_2016_all,
                     GMRT_2014_all,LOFAR_Patil_2017,PAPER_64_all]
    fig = p.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for result in published:
        print result.__name__
        data = result()
        for z in data:
            print z,data[z].shape
        min_data  = min_limit(data,**krange_table[result.__name__])
        redshifts = min_data.keys()
        #select off the UPPER LIMIT, regardless of whether it is or not.
        min_limits = np.array([min_data[z][2] for z in redshifts])
        ax.errorbar(redshifts,min_limits,min_limits/1.5,capsize=capsize,
            **formats[result.__name__])
    #load & plot the unpublished data
    if not files is None:
        unpublished_data = load_pspecs(files)
        for z in unpublished_data:
            print z,unpublished_data[z].shape
        min_data  = min_limit(unpublished_data,krange=krange)
        redshifts = sorted(min_data.keys())
        for _z in redshifts:
            print _z,min_data[_z][2], np.sqrt(min_data[_z][2])
        min_limits = np.array([min_data[z][2] for z in redshifts])
        ax.errorbar(redshifts,min_limits,min_limits/1.5,capsize=capsize,
            label=title,uplims=True,fmt='kd')

    ax.set_yscale('log')
    ax.set_ylabel('$\Delta^{2} (mK)^{2}$')
    ax.set_ylim([1e0, 1e9])
    ax.set_xlabel('z')
    ax.grid(axis='y')
    # Add model data.
    if models:
        if verbose:
            print 'Plotting 21cmFAST Model'
        simk = 0.2
        xlim = ax.get_xlim()  # save the data xlimits
        parm_array, k_array, delta2_array, delta2_err_array = load_andre_models()
        k_index = n.abs(k_array[0]-simk).argmin()
        alphaXs = n.sort(list(set(parm_array[:, 3])))
        Mmins = n.sort(list(set(parm_array[:, 4])))
        Nxs = n.sort(list(set(parm_array[:, 2])))
        for Nx in Nxs:
            for alphaX in alphaXs:
                for Mmin in Mmins:
                    _slice = n.argwhere(all_and([
                                        parm_array[:, 2] == Nx,
                                        parm_array[:, 3] == alphaX,
                                        parm_array[:, 4] == Mmin]
                                        ))

                    if len(_slice)==0:continue
                    if alphaX==0:
                        label='Cold Reionization'
                        ls = ':k'
                        fig.text(.65, .5, label, fontsize=10)
                    else:
                        label = "Fiducial 21cmFAST model"
                        ls = '-k'
                        fig.text(.65, .36, label, fontsize=10)
                    ax.plot(parm_array[_slice,0],
                            delta2_array[_slice,k_index], ls)#,label=label)
                #    ax.plot(parm_array[_slice, 0],
                #            delta2_array[_slice, k_index], '-k',
                #            label='Fiducal 21cmFAST model')
        ax.set_xlim(xlim)  # reset to the data xlimits

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if cnt > 0 else h for cnt, h in enumerate(handles)]
    num_hands = len(handles)
    # this array split makes it so that all MWA is on one line of the 
    # figure caption it looks a little weird as code but it's a hack
    # for now to get it look pretty
    #handles = np.array_split(handles, 2)
    #labels = np.array_split(labels, 2)
    #handles = list(itertools.chain.from_iterable(itertools.izip_longest(*handles)))
    #labels = list(itertools.chain.from_iterable(itertools.izip_longest(*labels)))
    #handles.insert(num_hands, handles.pop(-1))
    #labels.insert(num_hands, labels.pop(-1))
    ymax=[np.max(np.abs(_da)) for lines in ax.get_lines() for _da in lines.get_ydata() ]
    ymax = np.max(ymax)
    yticks = np.power(10, np.arange(0, np.log10(ymax)+1))
    ax.set_yticks(yticks)
    box = ax.get_position()
    ax.set_position([box.x0, box.height * .2 + box.y0,
                     box.width, box.height*.8])
    # fig.subplots_adjust(bottom=.275,top=.8)
    ax.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(.5, -.425), ncol=3, **kwargs)
    #ax.legend(handles, labels, loc='lower center',
    #          bbox_to_anchor=(.5, -.425), ncol=3, **kwargs)
    # ax.legend(loc='bottom',ncol=3)
    return fig



def plot_eor_summary(files=None, title='Input Files', k_mag=.2,
                     models=True, verbose=False, capsize=3.5, **kwargs):
    """Create summary plot of known EoR results.

    All inputs are optional.
    files: capo formated pspec k3pk files
    title: legend handle for input files
    k_mag: the k value to take the limits near (find limits close to k_mag)
    modles: boolean, plot fiducal 21cmfast models (included in capo)
    capsize: defines the size of the upper limits caps.
    verbose: print info while plotting
    """
    fig = p.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    # plot the GMRT paciga 2014 data
    GMRT = GMRT_2014_all()
    GMRT_results = {}
    if verbose:
        print('GMRT')
    for i, z in enumerate(GMRT.keys()):
        # index = n.argwhere(GMRT[z][:,0] - .2 < 0.1).squeeze()
        freq = cosmo_units.f21/(z+1)
        k_horizon = n.sqrt(cosmo_units.eta2kparr(30./c, z)**2 +
                           cosmo_units.u2kperp(15*freq*1e6/cosmo_units.c, z)**2)
        index = n.argwhere(abs(GMRT[z][:, 0] - k_mag)).squeeze()
        GMRT_results[z] = n.min(GMRT[z][index, 2])
        if verbose:
            print('results: {0},\t{1}'.format(z, n.sqrt(GMRT_results[z])))
        ax.errorbar(float(z), GMRT_results[z], GMRT_results[z]/1.5,
                    fmt='p', ecolor='gray', color='gray', uplims=True,
                    label='Paciga, 2013' if i == 0 else "",
                    capsize=capsize)

    # Get MWA 32 data
    MWA_results = {}
    MWA = MWA_32T_all()
    if verbose:
        print('Results: Z,\t Upper Limits')
        print('MWA 32')
    for i, z in enumerate(MWA.keys()):
        # index = n.argwhere(MWA[z][:,0] - .2 < .01).squeeze()
        freq = cosmo_units.f21/(z+1)
        k_horizon = n.sqrt(cosmo_units.eta2kparr(30./c, z)**2 +
                           cosmo_units.u2kperp(15*freq*1e6/cosmo_units.c, z)**2)
        index = n.argwhere(abs(MWA[z][:, 0] > k_horizon)).squeeze()
        MWA_results[z] = n.min(MWA[z][index, 2])
        if verbose:
            print('results: {0},\t{1}'.format(z, n.sqrt(MWA_results[z])))
        ax.errorbar(float(z), MWA_results[z], MWA_results[z]/1.5,
                    fmt='r*', uplims=True,
                    label='Dillon, 2014' if i == 0 else "",
                    capsize=capsize)

    MWA128_results = {}
    MWA128 = MWA_128_all()
    if verbose:
        print('MWA 128')
    for i, z in enumerate(MWA128.keys()):
        index = n.argmin(abs(MWA128[z][:, 0] - k_mag)).squeeze()
        MWA128_results[z] = n.min(MWA128[z][index, 2])
        if verbose:
            print('results: {0},\t{1}'.format(z, n.sqrt(MWA128_results[z])))
        ax.errorbar(float(z), MWA128_results[z], MWA128_results[z]/1.5,
                    fmt='y*', uplims=True, alpha=.5,
                    label='Dillon, 2015' if i == 0 else "", capsize=capsize)

    MWA_beards = MWA_128_beards()
    MWA_beards_results = {}
    if verbose:
        print('MWA Beardsley')
    for i, z in enumerate(MWA_beards.keys()):
        index = n.argmin(abs(MWA_beards[z][:, 0] - k_mag).squeeze())
        MWA_beards_results[z] = n.min(MWA_beards[z][index, 2])
        if verbose:
            print('results: {0},\t{1}'.format(z, n.sqrt(MWA_beards_results[z])))
        ax.errorbar(float(z), MWA_beards_results[z], MWA_beards_results[z]/1.5,
                    fmt='g*', uplims=True,
                    label='Beardsley, 2016' if i == 0 else "",
                    capsize=capsize)

    # Get Paper-32 data
    PSA32 = PAPER_32_all()
    PSA32_results = {}
    Jacobs_et_al = [0, 1, 2, 4]
    if verbose:
        print('PSA32')
    for i, z in enumerate(PSA32.keys()):
        index = n.argmin(abs(PSA32[z][:, 0] - k_mag)).squeeze()
        PSA32_results[z] = n.min(PSA32[z][index, 2])
        if verbose:
            print('results: {0},\t{1}'.format(z, n.sqrt(PSA32_results[z])))

        if i in Jacobs_et_al:
            ax.errorbar(float(z), PSA32_results[z], PSA32_results[z]/1.5,
                        fmt='md', uplims=True,
                        label='Jacobs, 2015' if i == 0 else "",
                        capsize=capsize)
        else:
            ax.errorbar(float(z), PSA32_results[z], PSA32_results[z]/1.5,
                        fmt='cv', uplims=True, label='Parsons, 2014',
                        capsize=capsize)

    # Get PAPER-64 results
    PSA64 = PAPER_64_all()
    PSA64_results = {}
    if verbose:
        print('PSA64')
    for z in PSA64.keys():
        index = n.argmin(abs(PSA64[z][:, 0] - k_mag)).squeeze()
        PSA64_results[z] = n.min(abs(PSA64[z][index, 2]))
        if verbose:
            print('results: {0},\t{1}'.format(z, n.sqrt(PSA64_results[z])))
        ax.errorbar(float(z), PSA64_results[z], PSA64_results[z]/1.5,
                    fmt='bs', uplims=True, label='Ali, 2015', capsize=capsize)

    # zs = [10.87,8.37]
    results = {}
    if verbose:
        print('Input files')
    zs, ks, k3pk, k3err = get_k3pk_from_npz(files)
    for i, z in enumerate(zs):
        results_array = n.array([ks[i], k3pk[i], k3pk[i] + k3err[i],
                                 k3pk[i] - k3err[i]]).T
        negs = n.argwhere(k3pk[i] < 0).squeeze()
        try:
            len(negs)
        except:
            negs = n.array([negs.item()])
        if len(negs) > 0:
            results_array[negs, -2], results_array[negs, -1] = abs(results_array[negs, -1]), -1 * results_array[negs, -2]
        index = n.argmin(abs(results_array[:, 0] - k_mag)).squeeze()
        results[z] = n.min(abs(results_array[index, 2]))
        if verbose:
            print('results: {0},\t{1}'.format(z, n.sqrt(results[z])))
        ax.errorbar(float(z), results[z], results[z]/1.5,
                    fmt='ko', uplims=True,
                    label=title if i == 0 else "",
                    capsize=capsize)

    ax.set_yscale('log')
    ax.set_ylabel('$\Delta^{2} (mK)^{2}$')
    ax.set_ylim([1e0, 1e7])
    ax.set_xlabel('z')
    ax.grid(axis='y')

    # Add model data.
    if models:
        if verbose:
            print 'Plotting 21cmFAST Model'
        simk = 0.2
        xlim = ax.get_xlim()  # save the data xlimits
        parm_array, k_array, delta2_array, delta2_err_array = load_andre_models()
        k_index = n.abs(k_array[0]-simk).argmin()
        alphaXs = n.sort(list(set(parm_array[:, 3])))
        Mmins = n.sort(list(set(parm_array[:, 4])))
        Nxs = n.sort(list(set(parm_array[:, 2])))
        for Nx in Nxs:
            for alphaX in alphaXs:
                for Mmin in Mmins:
                    _slice = n.argwhere(all_and([
                                        parm_array[:, 2] == Nx,
                                        parm_array[:, 3] == alphaX,
                                        parm_array[:, 4] == Mmin]
                                        ))
                    ax.plot(parm_array[_slice, 0],
                            delta2_array[_slice, k_index], '-k',
                            label='Fiducal 21cmFAST model')
        ax.set_xlim(xlim)  # reset to the data xlimits

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if cnt > 0 else h for cnt, h in enumerate(handles)]
    num_hands = len(handles)
    handles.insert(num_hands, handles.pop(0))
    labels.insert(num_hands, labels.pop(0))
    box = ax.get_position()
    ax.set_position([box.x0, box.height * .2 + box.y0,
                     box.width, box.height*.8])
    # fig.subplots_adjust(bottom=.275,top=.8)
    ax.legend(handles, labels, loc='lower center',
              bbox_to_anchor=(.5, -.425), ncol=3, **kwargs)
    # ax.legend(loc='bottom',ncol=3)
    return fig
