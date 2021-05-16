#! /usr/bin/env python
'''
Creates a summary plot of the Best data vs Redshift from  hera_sandbox.eor_results
'''
import matplotlib as mpl
from matplotlib import pyplot as p
import sys
sys.path.append('/home/mkolopanis/src/hera_sandbox/src/')
import eor_results
import sys
fontsize=15
legendfontsize=10
#mpl.rcParams['font.size']  = 20
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['legend.numpoints']  = 1
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = legendfontsize
mpl.rcParams['figure.dpi'] = 400
mpl.rcParams['savefig.dpi'] = 400
mpl.rcParams['savefig.format'] ='png'
mpl.rcParams['lines.markeredgewidth'] = 0
mpl.rcParams['lines.markersize'] = 9
#mpl.rcParams['lines.markersize'] = 7
published = [eor_results.MWA_32T_all,eor_results.LOFAR_Patil_2017,
             eor_results.MWA_128_all, eor_results.GMRT_2014_all,
             eor_results.MWA_128_beardsley_2016_all
             #eor_results.PAPER_32_parsons,
             #eor_results.PAPER_32_all,
             #eor_results.PAPER_64_all,
            ]

#published = [eor_results.MWA_32T_all,
#             eor_results.LOFAR_Patil_2017,
#             eor_results.PAPER_32_parsons,
#             eor_results.MWA_128_all,
#             eor_results.GMRT_2014_all,
#             eor_results.PAPER_32_all,
#             eor_results.MWA_128_beardsley_2016_all,
#             eor_results.PAPER_64_all,
#            ]
if len(sys.argv[1:]):
    fig = eor_results.plot_lowest_limits(files=sys.argv[1:],title='This Work',
                published=published,krange=(0.3,0.6))
else:
    fig = eor_results.plot_lowest_limits(published=published,krange=(0.1,0.6))

fig.savefig('eor_lowest_limits.pdf', format='pdf')
fig.savefig('eor_lowest_limits.png', format='png')
