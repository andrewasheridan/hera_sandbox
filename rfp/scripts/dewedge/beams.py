"""Module for simulating different types of beams."""

import numpy as np

class Beam1d:
    """Base class for constructing a 1-dimensional beam."""
    def __init__(
        self,
        thetas,
        freqs=0.15,
        ref_theta=0,
        width=0.2,
        mfreq=0.15,
        chromaticity=0,
        dtype=np.float32,
    ):
        """
        Set up the base parameters for defining a beam.

        Chromatic beams may be simulated by either providing an array of
        ``width`` parameters, or by specifying a chromaticity and reference
        frequency. When providing an array of frequencies along with a single
        width, reference frequency, and chromaticity, the beam width is
        modulated according to:

        :math:`\\sigma(\\nu) = \\sigma_0 (\\nu/\\nu_m)^\\alpha`,

        where :math:`\\alpha` is the chromaticity, :math:`\\nu_m` is the
        reference frequency, :math:`\\sigma_0` is the provided width, and
        :math:`\\nu` corresponds to the array of observed frequencies.
        
        Parameters
        ----------
        thetas: array-like of float
            Angular positions on the sky, in radians, with zero representing
            zenith.
        freqs: array-like of float, optional
            Frequencies at which to evaluate the beam, in GHz; only required
            for simulating chromatic beams.
        ref_theta: float, optional
            Direction the beam is pointed, in radians. Default is to assume
            a drift-scan telescope, with the peak response at zenith.
        width: float or array-like of float, optional
            ``Width'' of the beam in radians. For a Gaussian beam, this
            represents the standard deviation of the beam. Custom beam
            chromaticity may be simulated by passing an array of widths with
            the same shape as ``freqs``. Default is 0.2 radians.
        mfreq: float, optional
            Reference frequency, in GHz, for constructing chromatic beams if
            only a single width is provided. Default is 150 MHz.
        chromaticity: float, optional
            Spectral slope of the beam. For beams that increase in width with
            frequency, set this to a positive value; for beams that become
            more narrow with increasing frequency, set this to a negative value.
            Default is to not add any chromaticity to the beam.
        dtype: type, optional
            Data type to use for calculating the beam response. Default is to
            use 32-bit floats.
        """
        # Output will have shape Nthetas x Nfreqs
        self.thetas = np.atleast_2d(thetas).T
        self.freqs = np.atleast_2d(freqs)
        self.width = np.atleast_1d(width)
        self.horizon = np.pi / 2
        self.ref_theta = ref_theta
        self.mfreq = mfreq
        self.chromaticity = chromaticity
        self.dtype = dtype

    def gaussian(
        self,
        ref_theta=None,
        width=None,
        mfreq=None,
        chromaticity=None,
        dtype=None,
        power=True,
    ):
        """Simulate a Gaussian response.

        Parameters
        ----------
        ref_theta: float, optional
            Direction the beam is pointed, in radians. Default is to assume
            a drift-scan telescope, with the peak response at zenith.
        width: float or array-like of float, optional
            ``Width'' of the beam in radians. For a Gaussian beam, this
            represents the standard deviation of the beam. Custom beam
            chromaticity may be simulated by passing an array of widths with
            the same shape as ``freqs``. Default is 0.2 radians.
        mfreq: float, optional
            Reference frequency, in GHz, for constructing chromatic beams if
            only a single width is provided. Default is 150 MHz.
        chromaticity: float, optional
            Spectral slope of the beam. For beams that increase in width with
            frequency, set this to a positive value; for beams that become
            more narrow with increasing frequency, set this to a negative value.
            Default is to not add any chromaticity to the beam.
        dtype: type, optional
            Data type to use for calculating the beam response. Default is to
            use 32-bit floats.
        power: bool, optional
            Whether to return beam power or beam field (the latter can be
            negative). Default is to return beam power.

        Returns
        -------
        response: array-like of float
            Peak-normalized beam response evaluated at all directions and
            frequencies provided at class initialization. Has shape
            ``(self.thetas.size, self.freqs.size)``.
        """
        thetas, widths, dtype = self._process_args(
            ref_theta, width, mfreq, chromaticity, dtype
        )
        response = np.exp(-0.5 * (np.sin(thetas) / np.sin(widths)) ** 2)
        response = np.where(np.abs(self.thetas) <= self.horizon, response, 0)
        if power:
            response = response ** 2
        return response.astype(dtype)

    def sinc(
        self,
        ref_theta=None,
        width=None,
        mfreq=None,
        chromaticity=None,
        dtype=None,
        power=True,
    ):
        """Simulate a sinc response.

        Parameters
        ----------
        ref_theta: float, optional
            Direction the beam is pointed, in radians. Default is to assume
            a drift-scan telescope, with the peak response at zenith.
        width: float or array-like of float, optional
            ``Width'' of the beam in radians. For a Gaussian beam, this
            represents the standard deviation of the beam. Custom beam
            chromaticity may be simulated by passing an array of widths with
            the same shape as ``freqs``. Default is 0.2 radians.
        mfreq: float, optional
            Reference frequency, in GHz, for constructing chromatic beams if
            only a single width is provided. Default is 150 MHz.
        chromaticity: float, optional
            Spectral slope of the beam. For beams that increase in width with
            frequency, set this to a positive value; for beams that become
            more narrow with increasing frequency, set this to a negative value.
            Default is to not add any chromaticity to the beam.
        dtype: type, optional
            Data type to use for calculating the beam response. Default is to
            use 32-bit floats.
        power: bool, optional
            Whether to return beam power or beam field (the latter can be
            negative). Default is to return beam power.

        Returns
        -------
        response: array-like of float
            Peak-normalized beam response evaluated at all directions and
            frequencies provided at class initialization. Has shape
            ``(self.thetas.size, self.freqs.size)``.
        """
        thetas, widths, dtype = self._process_args(
            ref_theta, width, mfreq, chromaticity, dtype
        )
        response = np.sinc(0.5 * np.sin(thetas) / np.sin(widths))
        response = np.where(np.abs(self.thetas) <= self.horizon, response, 0)
        if power:
            response = response ** 2
        return response.astype(dtype)
        
    def _process_args(self, ref_theta, width, mfreq, chromaticity, dtype):
        """Helper function for modifying arguments."""
        if ref_theta is None:
            ref_theta = self.ref_theta
        if chromaticity is None:
            chromaticity = self.chromaticity
        thetas = self.thetas - ref_theta
        width = np.atleast_1d(width) or self.width
        mfreq = mfreq or self.mfreq
        dtype = dtype or self.dtype
        if width.size == 1:
            width = width * (self.freqs / mfreq) ** chromaticity

        return thetas, width, dtype
