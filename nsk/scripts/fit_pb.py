"""
fit_pb.py
=========

Construct and constrain a series of primary beam
components from measurements of source fluxes
across the sky.

----------------------
Nick Kern
nkern@berkeley.edu
Oct. 2018
"""
import argparse
import os
import sys
import shutil
import glob
import numpy as np
import scipy.stats as stats
from scipy import linalg as la
import matplotlib.pyplot as plt
import healpy as hlp
from opticspy import zernike as Zernike
import emcee


def zernike(m, n, rho=None, phi=None, Nside=None, nest=False, maxR=1.0):
    """
    Produce a 2D Zernike polynomial given its order m and n
    and rho [unit disk] and phi [radians] (polar coordinates).
    Alternatively, one can feed simply an Nside healpix resolution
    and it will return a healpix map.
    Uses the opticspy.zernike module.

    Parameters
    ----------
    m : integer
        m order of zernike polynomial Z^m_n

    n : integer
        n order of zernike polynomial Z^m_n

    rho : array-like
        Array of radial coordinates (unit-disk)

    phi : array-like
        Array of angular coordinates [radians] (unit-disk)

    Nside : integer
        Healpix map nside specification.

    Returns
    -------
    z : zernike polynomial of degree m & n
    """
    # assert either polar coords or healpix
    if rho is not None or phi is not None:
        assert Nside is None, "Cannot specify polar coords and healpix at the same time"
        assert rho is not None and phi is not None, "Must feed both rho and phi for polar coordinates"
    elif Nside is not None:
        assert rho is None and phi is None, "Cannot specify polar coords and healpix at the same time"
        Npix = hlp.nside2npix(Nside)
        rho, phi = hlp.pix2ang(Nside, np.arange(Npix), nest=nest, lonlat=False)
        phi[rho > np.pi/2] = np.nan
        rho[rho > np.pi/2] = np.nan

    # get Noll index
    j = (n * (n + 2) + m) // 2
    j = [1, 3, 2, 5, 4, 6, 9, 7, 8, 10,
         15, 13, 11, 12, 14, 21, 19, 17,
         16, 18, 20, 27, 25, 23, 22, 24,
         26, 28, 35, 33, 31, 29, 30, 32,
         34, 36, 45, 43, 41, 39, 37, 38][j]

    # get polynomial
    Z = Zernike.Coefficient(**{"Z{}".format(j):1.0})

    # evaluate polynomial
    z = Zernike.__zernikepolar__(Z.__coefficients__, rho, phi)

    return -z

def pca_decompose(data, cov_estimator=np.cov):
    """
    Perform a PCA (aka eigenvector or KL) decomposition of
    training set data.

    

    """
    # Find Covariance
    Dcov = cov_estimator(data.T) 

    # Solve for eigenvectors and values using SVD
    u, eig_vals, eig_vecs = la.svd(Dcov)

    # Sort by eigenvalue
    eigen_sort = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[eigen_sort]
    eig_vecs = eig_vecs[eigen_sort]

    return eig_vals, eig_vecs


def lnLike(w, y, Phi, invcov=None):
    """
    Create a log-likelihood function between measurements (y)
    and a set of principal components (Phi) and their weights (w).

        lnL = -0.5 (y-m)^T invcov (y-m)

    where

          m = Phi * w

    Parameters
    ----------
    y : array-like


    Phi : array-like


    w : array-like


    invcov : array-like

    Returns
    -------
    lnL : float
        Value of the log-likelihood
    """
    # reshape w and y into column vectors
    if w.ndim == 1:
        w = w[:, None]
    if y.ndim == 1:
        y = y[:, None]

    # Evaluate m
    m = np.dot(Phi, w)

    # Get residual
    r = y - m

    # Get invcov
    if invcov is None:
        invcov = np.eye(len(m))

    # Evaluate lnL
    lnL = -0.5 * np.float(r.T.dot(invcov).dot(r))

    return lnL

def flat_lnPrior(w, bounds=[-1, 1], index=0):
    """
    Evaluate a flat 1D log-prior function on a single
    element of vector w specified by index.

    """
    pass

def gauss_lnPrior(w, mean=0, var=1, index=0):
    """
    Evaluate a Gaussian 1D log-prior function on a single
    element of vector w specified by index.

    """
    pass

def lnPost(w, y, Phi, invcov=None, prior_funcs=[]):
    """
    Generate an un-normalized log-Posterior function
    given measurements (y), principal components (Phi)
    and weights (w) to form a log-likelihood, and optionally
    any prior distributions on the elements of w: otherwise
    assumed to be flat priors.

    """

    # Evaluate log-likelihood
    lnL = lnLike(w, y, Phi, invcov=invcov)

    # Evaluate log-priors
    lnPr = 0
    if len(prior_funcs) > 0:
        lnPr += reduce(operator.add, [p(w) for p in prior_funcs])

    # Return log-posterior
    return lnL + lnPr

def run_ES(start_pos, lnPost, Nstep=1000, Nburn=0, a=2.0, args=[], kwargs={}):
    """
    Setup and run an Ensemble Sampler from emcee.


    """
    # setup sampler
    nwalkers, ndim = start_pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnPost, a=a, args=args, kwargs=kwargs)

    # burn in
    if Nburn > 0:
        pos, prob, state = sampler.run_mcmc(start_pos, Nburn)
        sampler.reset()
    else:
        pos = start_pos

    # explore
    sampler.run_mcmc(pos, Nstep)

    return sampler

