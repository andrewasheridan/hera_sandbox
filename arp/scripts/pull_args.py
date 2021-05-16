#! /usr/bin/env python
import sys, os
args = sys.argv[1:]
m = int(os.environ['NJOBS'])
i = int(os.environ['PBS_ARRAYID']) - 1
print ' '.join(args[i::m])
