import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import pylab as pl
from scipy.stats import linregress
from scipy.optimize import least_squares
import os

rms = False

cmap = pl.get_cmap('tab10')
#colors = [cmap(i) for i in range(len(Ncpus))]

for fname in sys.argv[1:]:

    f = np.load(fname)

    variances = f['variances']
    Nsamps = f['avg_lens_nsamp']
    avg_times = f['avg_lens_time']
    stds = f['var_stds']
    bl_labels = ["{}_{}".format(*t) for t in f['bls']]
    
    xvar = f['avg_lens_nsamp']
    xvar = xvar.astype(float)
    
    ## Using linregress:
    #linfit = linregress(np.log10(xvar), np.log10(variances))
    #b0 = linfit.slope
    #a0 = 10**(linfit.intercept)
    #fitted = a0*xvar**b0
    #print('a = {:.3f}, b = {:.3f}'.format(a0,b0))
    
    def fit_func(x, a, b):
        """
            a = Variance * averaging time
            b = power
            x = averaging time
        """
        #return variances[0]/xvar[0]**b * (x.astype(float))**b 
        return a*(x.astype(float))**(b)
    
    def residual(fit, time_arr, var, weight, c):
        a, b = fit
    #    c = int(c)
        return (np.log10(var[c:]) - np.log10(fit_func(time_arr[c:], a, b)))#*np.sqrt(float(c))#*weight
        #return (var - fit_func(time_arr, a,b))#*weight
    
    
    
    #TODO Define weights properly
    weights = f['nsamps'].astype(float)          #The more samples of variance, the more robust.
    
    guess = (np.max(variances)*xvar[0]**(-1), -1)
    #costs = []
    #cutmax = 500
    #for cutoff in range(0, cutmax):
    #    lsq_res = least_squares(residual, guess, args=(xvar.astype(float), variances, weights, cutoff), method='trf')#, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    #    print cutoff, lsq_res.cost
    #    costs.append(lsq_res.cost)
    if len(variances.shape) == 1:
        variances = variances[:, np.newaxis]
        stds = stds[:, np.newaxis]

    if rms:
        variances = np.sqrt(variances)/np.mean(f['avgs'],axis=(1,3)) * 100.0
        stds = np.sqrt(stds)/np.mean(f['avgs'],axis=(1,3)) * 100.0

    cutoff=0
    lsq_res = least_squares(residual, guess, args=(xvar.astype(float), variances[:,0], weights, cutoff), method='trf')
    fitted = fit_func(xvar, *lsq_res.x)
    print(np.mean(lsq_res.fun), lsq_res.x)
    print("fname: {}, Percent error with {:.1f} hours of averaging: {:.4f}%".format(fname, avg_times[-1] / 60., np.abs(np.sqrt(variances[-1,0])/np.mean(f['avgs'][-1,0]) )*100.0))

    Nbls = variances.shape[1]
    fig, ax = pl.subplots(1,1, dpi=200)
    for bl, label in enumerate(bl_labels):
#        col = str(bl/float(Nbls+2))
        col = cmap(bl)
        ax.fill_between(avg_times, (variances-stds)[:,bl], (variances+stds)[:,bl], alpha=0.5, color=col)
#        if bl == 0: ax.plot(avg_times, fitted, label='Lsq fit')
        label='Variance bl: '+label
        if rms: label = 'rms bl: '+label
        ax.plot(avg_times, variances[:,bl], label=label, color=col)
    ax.set_title(fname)
    ax.set_yscale('log')
    #ax.set_xscale('log')
    pl.legend()
    fname='.'.join(fname.split('.')[:-1])
    pl.savefig(fname+"_im.png")

#pl.show()

