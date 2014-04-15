#! /usr/bin/env python

"""
Add a row to the pdb.observations table and populate the pdb.files tables accordingly.

DFM
"""

from initDB import pdb
import sys
import re

hostname = None
list_of_jds = []

def file2jd(zenuv):
    return re.findall(r'\d+\.\d+', zenuv)[0]

for arg in sys.argv[1:]:
    if 'zen.245' in arg:
        jd = file2jd(arg)
        if not jd in list_of_jds:
            list_of_jds.append(jd)
    else:
        hostname = arg
list_of_jds.sort()

#exit if the host is bad.
if not pdb.has_record('hosts',hostname):
    sys.exit(1)

for i,jd in enumerate(list_of_jds):
    obscols = {}
    obscols['JD'] = jd
    for pi in 'xy':
        for pj in 'xy':
            obscols[pi+pj] = "zen.%s.%s.uv"%(jd,pi+pj)
            # add an entry to pdb.files here. Otherwise observations will break.
            filecols = {}
            filecols['JD'] = jd
            filecols['filename'] = obscols[pi+pj]
            filecols['basefile'] = obscols[pi+pj]
            filecols['host'] = hostname
            filecols['created_on'] = "NOW()"
            filecols['last_modified'] = "NOW()"
            pdb.addrow('files',filecols)
    try:
        obscols['jd_hi'] = list_of_jds[i+1]
    except(IndexError):
        pass
    try:
        obscols['jd_lo'] = list_of_jds[i-1]
        print
    except(IndexError):
        pass

    obscols['created_on'] = "NOW()"
    obscols['last_modified'] = "NOW()"

    pdb.addrow('observations',obscols)
