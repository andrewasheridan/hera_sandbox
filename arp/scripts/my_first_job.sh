#! /bin/bash
#PBS -N my_first_job
#PBS -q hera
#PBS -l nodes=1:ppn=4
#PBS -l walltime=24:00:00
#PBS -l vmem=10g
#PBS -j oe
#PBS -o my_first_job.out 
#PBS -V
#PBS -m be
#PBS -M aparsons@berkeley.edu
#PBS -t 1-10
export NJOBS=10
./pull_args.py 1 2 3 4 5 6 7 8 9 10 11 12
