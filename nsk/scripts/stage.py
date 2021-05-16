#!/usr/bin/env python2.7
"""
stage.py
--------

copy files from herastore
to local directory

Nicholas Kern
Jan. 2018
"""
import os
import sys
import glob
import argparse
import shutil

args = argparse.ArgumentParser(description="")

args.add_argument("search", type=str, help="glob-parsable search string")
args.add_argument("--jd", type=str, help="julian date string")
args.add_argument("--stage_dir", type=str, help="staging directory")
args.add_argument("--overwrite", default=False, action='store_true', help='overwrite staged files if they exist')

if __name__ == "__main__":
    a = args.parse_args()
    stage_dir = os.path.join(a.stage_dir, a.jd)
    print("stage_dir is {}".format(stage_dir))

    # check for staging directory
    if os.path.exists(stage_dir) is False:
        print("making stage_dir {}".format(stage_dir))
        os.mkdir(stage_dir)

    def copy_files(cp_files):
        for i, f in enumerate(cp_files):
            fbase = os.path.basename(f)
            fnew = os.path.join(stage_dir, fbase)
            if os.path.exists(fnew) and a.overwrite is False:
                print("{} exists, not overwriting".format(fnew))
                continue
            elif os.path.exists(fnew):
                print("{} exists, overwriting".format(fnew))
                try:
                    shutil.rmtree(fnew)
                except:
                    os.remove(fnew)
            else:
                print("copying {}".format(fnew))
            if os.path.isdir(f):
                shutil.copytree(f, fnew)
            elif os.path.isfile(f):
                shutil.copyfile(f, fnew)
            os.chmod(fnew, 0744)

    # search herastore01-1
    cp_files = sorted(glob.glob("/export/hera/herastore01-1/{}/{}".format(a.jd, a.search)))
    copy_files(cp_files)

    # search herastore01-2
    cp_files = sorted(glob.glob("/export/hera/herastore01-2/{}/{}".format(a.jd, a.search)))
    copy_files(cp_files)

    # search herastore01-3
    cp_files = sorted(glob.glob("/export/hera/herastore01-3/{}/{}".format(a.jd, a.search)))
    copy_files(cp_files)
    
    # search herastore01-4
    cp_files = sorted(glob.glob("/export/hera/herastore01-4/{}/{}".format(a.jd, a.search)))
    copy_files(cp_files)
    



