"""Helpful functions for de-wedging research."""

import itertools
import numpy as np

from astropy import constants

def get_coverage(antpos, freqs, bin_edges=None, mode="u"):
    """
    Determine the number of baselines that sample each mode.

    Unless bin edges for an array of u(vw)-modes is provided, this will count
    integer u(vw)-modes that are sampled by the array.

    Parameters
    ----------
    antpos: dict
        Dictionary mapping antenna numbers to ENU baselines in meters.
    freqs: array-like of float
        Array of frequencies in GHz.
    bin_edges: array-like of float or dict, optional
        Bin edges corresponding to gridding of the uvw coordinate system. If
        passing a dictionary, keys should come from the set {'u', 'v', 'w'},
        and values should be arrays of floats specifying bin edges.
        Default is to use unity-length bins, centered on integer values.
    mode: str, optional
        Whether to bin in u, v, w, or some combination of the three. Default
        is to bin in u.

    Returns
    -------
    coverage: dict
        Dictionary mapping mode (one of 'u', 'v', 'w') to counts per bin. Keys
        are the characters in the ``mode`` string, values are the counts per bin
        for each mode.
    bin_edges: dict
        Dictionary mapping mode to bin edges used for that mode.
    """
    # Make sure bin edges are broadcastable to modes for which to get coverage.
    uvw_to_ind = dict(zip('uvw', range(3)))
    baselines = get_baselines(antpos)
    if bin_edges is not None:
        if not isinstance(bin_edges, dict):
            try:
                bin_edges = np.array(bin_edges).astype(np.float)
                if bin_edges.ndim not in (1, 2): raise ValueError
            except ValueError:
                raise ValueError("Bin edges could not be parsed.")
            if bin_edges.ndim == 2:
                if bin_edges.shape[0] != len(mode):
                    raise ValueError(
                        "2-D arrays of bin edges must have the same number "
                        "of rows as modes specified. You provided bin edges "
                        f"with shape {bin_edges.shape}, but want to calculate "
                        f"coverage for {len(mode)} modes ({set(mode)})."
                    )
                bin_edges = dict(zip(mode, bin_edges))
            else:
                # We won't be modifying the bin edges, so this is safe.
                bin_edges = dict.fromkeys(mode, bin_edges)
        else:
            if set(bin_edges.keys()) != set(mode):
                raise ValueError("bin_edges keys do not match provided modes.")
    else:
        mode_bounds = {
            m: (
                min(bl[uvw_to_ind[m]] for bl in baselines.values()),
                max(bl[uvw_to_ind[m]] for bl in baselines.values())
            )
            for m in mode
        }
        for m, bounds in mode_bounds.items():
            if bounds[0] < 0:
                lower_bound = bounds[0] * freqs.max()
            else:
                lower_bound = bounds[0] * freqs.min()
            if bounds[1] < 0:
                upper_bound = bounds[1] * freqs.min()
            else:
                upper_bound = bounds[1] * freqs.max()
            mode_bounds[m] = (
                int(np.floor(lower_bound)) - 1,
                int(np.ceil(upper_bound)) + 1
            )
        
        # Ensure bin centers are integers from lower_bound to higher_bound.
        bin_edges = {m: np.arange(*bounds) + 0.5 for m, bounds in mode_bounds.items()}

    coverage = {m: np.zeros(len(bins) - 1) for m, bins in bin_edges.items()}
    for bl in baselines.values():
        for m, bins in bin_edges.items():
            hist = np.histogram(freqs * bl[uvw_to_ind[m]], bins=bins)[0]
            # Each baseline can only count once, so don't overcount.
            hist = np.min(np.vstack([np.ones(hist.size), hist]), axis=0)
            coverage[m] += hist
            
    return coverage, bin_edges


def get_baselines(antpos, autos=False):
    """Construct dictionary mapping antenna pairs to baselines."""
    if autos:
        combinations = itertools.combinations_with_replacement
    else:
        combinations = itertools.combinations
    return {
        pair: np.array(antpos[pair[1]]) - np.array(antpos[pair[0]])
        for pair in combinations(antpos.keys())
    }
