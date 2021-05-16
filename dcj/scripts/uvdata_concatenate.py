#! /usr/bin/env python
"""
Use pyuvdata to concatenate a list of files
Reads miriad, writes uvfits
"""
import argparse
import pyuvdata
from memory_profiler import profile


a = argparse.ArgumentParser()


a.prog = 'uvdata_concatenate.py'
a.add_argument('infiles', metavar='infiles', nargs='*', type=str, default=[],
                       help='input files to concatenate')
a.add_argument('-O', '--outputfile', metavar='outputfile',type=str,default='merged.uvfits',
                       help='output file to write')

args = a.parse_args()
@profile
def uvdata_concat(files,outputfile):
    UV = pyuvdata.uvdata.UVData()
    print("reading {n} files".format(n=len(files)))
    UV.read_miriad(files)
    print("writing: {f}".format(f=outputfile))
    UV.write_uvfits(outputfile,force_phase=True,spoof_nonessential=True)
uvdata_concat(args.infiles,args.outputfile)
