# ML experiments for HERA RedCal

Can we use ML to aid redcal? Probably. We use tensorflow to find out.

See `wrap_unwrap/wrap_unwrap_initial_exploration.ipynb` for a brief overview of the problem of computing cable delays from complex data.

The majority of the work here is trying to find cable delays using ML.

## Prereqs

```
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

TODO: Add description of how data is manipulated. For now see `modules/data_manipulation.py`

## Directory Contents
`zen_data/` - data - not in repo

`wrap_unwrap/` - quick overview

`modules/` - data manipulation & creators, base classes. 

`solver/` - solver for Ax = y using the networks output (non functional)

`fully_connected_network/` - one type of deep fully connected network

`convolutional_network/` - multiple convolutional networks of various configurations. Current focus

TODO: clean up directory structure ?

## Current Focus
 - mostly working in `convolutional_network/` on `CNN_DS_BN_C.py` & `CNN_C.ipynb`
  - see `convolutional_network/_Convolutional_Networks.ipynb` for a description