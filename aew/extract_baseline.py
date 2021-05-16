import glob
import sys
import numpy as np
from pyuvdata import UVData
import pyyaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', dest = 'config')
parser.add_argument('--output', '-o', dest = 'output')
parser.add_argument('--ddir', '-d', dest='ddir')

parser.parse_args()
config = parser.config
rdir = parser.ddir


config = sys.argv[1]
rdir = sys.argv[1]
ant1 = int(sys.argv[2])
ant2 = int(sys.argv[3])
chmin = int(sys.argv[4])
chmax = int(sys.argv[5])

files = glob.glob(rdir + '/*.HH.uvh5')
jds = []
for fname in files:
    fname_jd = fname.split('/')[-1]
    jd = float(re.findall('[0-9]{7}.[0-9]{5}',fname_jd)[0])
    jds.append(jd)
jds = np.array(jds)

files = list(np.array(files)[np.argsort(jds)])
jds = jds[np.argsort(jds)]
files = list(np.array(files)[np.argsort(jds)])

files_diff = []
for jd in jds:
    files_diff.append(glob.glob('%s/*%s*.HH.diff.uvh5'%str(ddir,jd))[0])

uvd = UVData()
uvd.read_uvh5(files,bls = [(ant1,ant2)],freq_chans = np.arange(chmin,chmax).astype(int))

uvd_diff = UVData()
uvd_diff.read_uvh5(files_diff,bls=[(ant1,ant2)],freq_chans = np.arange(chmin,chmax).astype(int)))

uvd.write_uvh5('%d.HH.a1_%d_a2_%d_ch_%d_%d.uvh5'%(jd,ant1,ant2,chmin,chmax)))
uvd_diff.write_uvh5('%d.HH.diff.a1_%d_a2_%d_ch_%d_%d.uvh5'%(jd,ant1,ant2,chmin,chmax)))
