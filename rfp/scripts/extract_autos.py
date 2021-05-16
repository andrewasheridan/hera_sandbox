# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 the HERA Collaboration
# Licensed under the 2-clause BSD license.
import glob
import os
import re
import sys

from pyuvdata import UVData

# figure out which directory the data lives in
data_dir = sys.argv[1]
save_dir = sys.argv[2]
pattern = re.compile("[0-9]{7}")
jd = pattern.findall(data_dir)[0]
if jd in ('2458790', '2458791', '2458792', '2458799'):
    data_dir = os.path.join(data_dir, jd)
file_glob = sorted(
    glob.glob(os.path.join(data_dir, "zen.*.uvh5"))
)
sum_file_glob = list([fname for fname in file_glob if "diff" not in fname])
diff_file_glob = list([fname for fname in file_glob if "diff" in fname])
if len(file_glob) == 0:
    raise FileNotFoundError(
        "Something went wrong--no files were found."
    )

# Load in the autos for the sum/diff files and write to disk.
for file_glob, file_type in zip([sum_file_glob, diff_file_glob], ('sum', 'diff')):
    uvd = UVData()
    uvd.read(file_glob, axis='blt', ant_str='auto')
    save_filename = os.path.join(
        save_dir, f"zen.{jd}.autos.{file_type}.uvh5"
    )
    uvd.write_uvh5(save_filename, clobber=True)

