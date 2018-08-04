# ML experiments for HERA RedCal

Can we use ML to aid redcal? Probably. We use tensorflow to find out.

See `wrap_unwrap/wrap_unwrap_initial_exploration.ipynb` for a brief overview of the problem of computing cable delays from complex data.

The majority of the work here is trying to find cable delays using ML.

## Prereqs

```
tensorflow
pyuvdata
hera_cal

# for blue percepturally uniform colormap
# see https://github.com/bokeh/colorcet
colorcet (conda install -c bokeh colorcet)
``` 

- and all their prereqs

## Data

Data is built with NRAO IDR-2 miriad / firstcal files (is this description correct?).

 Specific files used cover one ten minute window:
`zen_data/zen.2458098.58037.xx.HH.uv` &
`zen_data/zen.2458098.58037.xx.HH.uv.abs.calfits`

TODO: Add description of how data is manipulated. For now see `modules/data_manipulation.py`

## Directory Contents

`data/` - generated data - not in repo

`data_creation/` - how the data is generated from raw data

`experiments/` - Experiments in training different networks, solving Ax = b, and doing other things.

`modules/` - base classes, helpers. 

`network_trainers/` - classes for training different network types

`networks/` - various different network types

`zen_data/` - raw data - not in repo

`wrap_unwrap/` - quick overview of the wrap unwrap problem


## Current Focus
 - mostly working in `experiments/` on `CNN_DS_BN_C.py` & `CNN_C.ipynb`
  - see `networks/Network_Info.ipynb` for a description
