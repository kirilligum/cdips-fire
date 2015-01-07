#!/bin/bash
qsub -V -q regular -A m2063 -l mppwidth=24 -l walltime=05:58:58 -N "$1$2$3" <<< " 
cd $PWD
module unload python
module load anaconda
export OMP_NUM_THREADS=24
pwd
aprun -n 1 -N 1 -d 24 time python $@"
