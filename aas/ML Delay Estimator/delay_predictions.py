import matplotlib.pyplot as plt
import numpy as np


def gaussian(x, mu, sigma,A):
    return A*((2.*np.pi*sigma**2)**(-1./2.))*np.exp(-((x-mu)**2/(2*sigma**2)))

from scipy.optimize import curve_fit
    
class delay_prediction(object):
    def __init__(self, data, bins = 10):
        
        self.data = data
        self.bins = bins
        self._data_std = np.std(self.data)
        self._data_mean = np.mean(self.data)

        entries, edges = np.histogram(data, bins = self.bins)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        
        try:
            popt, _ = curve_fit(gaussian,bin_centers,entries, p0 = [self._data_mean, self._data_std, max(entries)])
            self.fit_mu  = popt[0]
            self.fit_sig = popt[1]
            self.fit_amp = popt[2]
            
        except:
            self.fit_mu  = None
            self.fit_sig = None
            self.fit_amp = None 
            
    def plot(self):
        
        fig, ax = plt.subplots()
        ax.hist(self.data, bins = self.bins, color = 'steelblue')
        
        if self.fit_mu:
            x_range = np.linspace(self.fit_mu - 3*self.fit_sig,self.fit_mu + 3*self.fit_sig, 100)
            ax.plot(x_range, gaussian(x_range, self.fit_mu, self.fit_sig, self.fit_amp), color = '#E50000')
            ax.set_xticks(np.arange(self.fit_mu - 3*self.fit_sig, pt.fit_mu + 4*self.fit_sig, self.fit_sig))
            plt.setp(ax.get_xticklabels()[::2], visible=False)

            ax.text(0.88,0.9,r'$\mu = {:>2.4}$'.format(self.fit_mu), transform = ax.transAxes, horizontalalignment = 'center')
            ax.text(0.88,0.85,r'$\sigma = {:>2.4}$'.format(self.fit_sig), transform = ax.transAxes, horizontalalignment = 'center')
        else:

            ax.text(0.12,0.9,r'$\bar{\tau} = %.5f$' % self._data_mean, transform = ax.transAxes, horizontalalignment = 'center')
            ax.text(0.12,0.85,r'$\tau_{\sigma} = %.5f$' % self._data_std, transform = ax.transAxes, horizontalalignment = 'center')

        plt.show()