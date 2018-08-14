# Estimating Delays
## ML experiments for HERA RedCal

Can we use ML to aid redcal? We use tensorflow to find out.

See `wrap_unwrap_initial_exploration.ipynb` for a brief overview of the problem of computing cable delays from complex data.

## Prereqs

```
numpy
matplotlib
tensorflow
pyuvdata
hera_cal
```
- and all their prereqs

## Data

Data is built with NRAO IDR-2 miriad / firstcal files (is this description correct?).

 Specific files used cover one ten minute window:
`zen_data/zen.2458098.58037.xx.HH.uv` &
`zen_data/zen.2458098.58037.xx.HH.uv.abs.calfits`

TODO: Add description of how data is manipulated. For now see `estdel/nn/data_manipulation.py`

## Directory Contents

`estdel/` - package for estimating delays

`experiments/` - Experiments in training different networks, solving Ax = b, and doing other things.