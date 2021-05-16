'''
This script preprocesses autocorrelations from h2c.
Given a list of observations
---------------------------------------------------
1) it runs xrfi on largest manageable chunk of data in files from a directory.
2) writes intermediary flagged miriad files for each chunk (autocorrs only)
3) concatenates all chunks into a single file.
4) delete intermediary outputs
----------------------------------------------------
'''

import numpy as np
from hera_qm import xrfi
from pyuvdata import UVData
import argparse
import os
import re
import glob
import shutil
cwd = os.getcwd()

parser = argparse.ArgumentParser(description='Generate rfi flagged autocorrelations using all available data at each time.')
parser.add_argument('-d','--directory',dest='directory')
#parser.add_argument('-f','--filelist',dest='filelist')
parser.add_argument('-c','--chunk_size',dest='chunk_size',default=1)
parser.add_argument('--outputdir',dest='outputdir',default=cwd)
parser.add_argument('--tempdir','--temp',dest='tempdir',default=cwd)
parser.add_argument('--output','-o',dest='output')
parser.add_argument('--freq_threshold','-f',dest='freq_threshold',default=0.5)
parser.add_argument('--time_threshold','-t',dest='time_threshold',default=0.05)
parser.add_argument('--clobber',dest='clobber',default=True)
parser.add_argument('--cleanup',dest='cleanup',default=True)
parser.add_argument('--fmin','-l',dest='fmin',default=None)
parser.add_argument('--fmax','-u',dest='fmax',default=None)
parser.add_argument('--onlyautoflags','-a',dest='onlyautoflags',default=False)
parser = parser.parse_args()
#user can specify a list of files or
#assert not and(parser.filelist, parser.directory)

chunk_size =int(parser.chunk_size)
dir = parser.directory
tempdir = parser.tempdir
outputdir = parser.outputdir
output = parser.output
freq_threshold = float(parser.freq_threshold)
time_threshold = float(parser.time_threshold)
clobber = bool(parser.clobber)
cleanup=bool(parser.cleanup)
onlyautoflags = bool(parser.onlyautoflags)

if not parser.fmin is None:
    fmin = float(parser.fmin)
else:
    fmin = parser.fmin
if not parser.fmax is None:
    fmax = float(parser.fmax)
else:
    fmax = parser.fmax

print(freq_threshold)
print(time_threshold)
files = glob.glob(dir+'/*.HH.uvh5')
jds = []
for fname in files:
    fname_jd = fname.split('/')[-1]
    jd = float(re.findall('[0-9]{7}.[0-9]{5}',fname_jd)[0])
    jds.append(jd)
jds = np.array(jds)
files = list(np.array(files)[np.argsort(jds)])
jds = jds[np.argsort(jds)]
files = list(np.array(files)[np.argsort(jds)])
#get diff files
files_diff = []
for jd in jds:
    files_diff.append(glob.glob(dir+'/*%s*.HH.diff.uvh5'%str(jd))[0])
chunks = int(len(files)/chunk_size) + 1
file_chunks = [files[cnum*chunk_size:(cnum+1)*chunk_size] for cnum in range(chunks-1)]
file_chunks_diff = [files_diff[cnum*chunk_size:(cnum+1)*chunk_size] for cnum in range(chunks-1)]
jd_chunks = [jds[cnum*chunk_size:(cnum+1)*chunk_size] for cnum in range(chunks-1)]

file_chunks = file_chunks + [files[chunk_size*(chunks-1):]]
file_chunks_diff = file_chunks_diff + [files_diff[chunk_size*(chunks-1):]]
jd_chunks = jd_chunks + [jds[chunk_size*(chunks-1):]]
print(chunks)
print(file_chunks)
print(file_chunks_diff)
print(jd_chunks)
chunk_files = []
chunk_files_diff = []

for cnum,chunk,chunk_d in zip(range(chunks),file_chunks,file_chunks_diff):
    uv = UVData()
    uv.read_uvh5(chunk)
    uvd = UVData()
    uvd.read_uvh5(chunk_d)
    if fmin is None:
        fmin = np.min(uv.freq_array)
    if fmax is None:
        fmax = np.max(uv.freq_array)

    freq_select = np.logical_and(uv.freq_array[0]<=fmax, uv.freq_array[0]>=fmin)
    nf = len(freq_select)

    uv.select(frequencies=uv.freq_array[0][freq_select])
    uvd.select(frequencies=uvd.freq_array[0][freq_select])
    print('fmax=%e'%uv.freq_array.max())
    print('fmin=%e'%uv.freq_array.min())
    if onlyautoflags:
        uv.select(bls = [(a,a) for a in np.unique(uv.ant_1_array)])
        uvd.select(bls = [(a,a) for a in np.unique(uv.ant_1_array)])

    # first round of flagging
    uvf_m, uvf_f = xrfi.xrfi_pipe(uv)
    #print(uvf_f.flag_array.shape)
    # apply first round
    xrfi.flag_apply(uvf_f, uv, keep_existing=False,force_pol=True)
    print(float(len(uv.flag_array[uv.flag_array]))/len(uv.flag_array.flatten()))
    # Second round
    uvf_m2, uvf_f2 = xrfi.xrfi_pipe(uv, alg='detrend_meanfilt')
    # Threshold
    uvf_temp = uvf_f2.copy()
    uvf_temp.to_metric(convert_wgts=True)
    uvf_final = xrfi.flag(uvf_temp, nsig_p=1.0, nsig_f=freq_threshold, nsig_t=time_threshold)
    xrfi.flag_apply(uvf_final,uv,force_pol=True)
    xrfi.flag_apply(uvf_final,uvd,force_pol=True)

    if not onlyautoflags:
        uv.select(bls = [(a,a) for a in np.unique(uv.ant_1_array)])
        uvd.select(bls = [(a,a) for a in np.unique(uv.ant_1_array)])


    chunk_files.append(tempdir+'/%s.%d.HH.temp.uvh5'%(output,cnum))
    chunk_files_diff.append(tempdir+'/%s.diff.%d.HH.temp.uvh5'%(output,cnum))
    uv.write_uvh5(str(chunk_files[-1]),clobber=True)
    uvd.write_uvh5(str(chunk_files_diff[-1]),clobber=True)

uv = UVData()
uvd = UVData()
uv.read(chunk_files)
uvd.read(chunk_files_diff)
uv.write_uvh5(outputdir+'/'+output+'.uvh5',clobber = clobber)
uvd.write_uvh5(outputdir+'/'+output+'.diff.uvh5',clobber = clobber)
#clean up temp files
if cleanup:
    for temp,temp_d in zip(chunk_files,chunk_files_diff):
        os.remove(temp)
        os.remove(temp_d)
