"""
ISGRI Data Quality Metrics
===========================

Statistical quality metrics for INTEGRAL/ISGRI light curves.

The main metric is reduced chi-squared (chisq/dof), which tests whether
count rates are consistent with Poisson statistics. Values near 1.0
indicate stable background and no variable sources.

Classes
-------
QualityMetrics : Compute chi-squared metrics for light curves

Examples
--------
>>> from isgri.utils import LightCurve, QualityMetrics
>>>
>>> # Load light curve
>>> lc = LightCurve.load_data("events.fits")
>>>
>>> # Compute quality metrics
>>> qm = QualityMetrics(lc, binsize=1.0, emin=20, emax=100)
>>> chi = qm.raw_chi_squared()
>>> print(f"Raw chisq/dof = {chi:.2f}")
>>>
>>> # Sigma-clipped (removes outliers)
>>> chi_clip = qm.sigma_clip_chi_squared(sigma=3.0)
>>> print(f"Clipped chisq/dof = {chi_clip:.2f}")
>>>
>>> # GTI-filtered (only good time intervals)
>>> chi_gti = qm.gti_chi_squared()
>>> print(f"GTI chisq/dof = {chi_gti:.2f}")

"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Union
from .lightcurve import LightCurve


class QualityMetrics:
    """
    Compute statistical quality metrics for ISGRI light curves.

    Uses module-by-module light curves to compute chi-squared statistics.
    Results are weighted by total counts per module.

    Parameters
    ----------
    lightcurve : LightCurve, optional
        LightCurve instance to analyze
    binsize : float, default 1.0
        Bin size in seconds
    emin : float, default 1.0
        Minimum energy in keV
    emax : float, default 1000.0
        Maximum energy in keV
    local_time : bool, default False
        If True, use local time (seconds from T0).
        If False, use IJD time. GTIs are always in IJD.

    Attributes
    ----------
    module_data : dict or None
        Cached rebinned data {'time': array, 'counts': array}

    Examples
    --------
    >>> lc = LightCurve.load_data("events.fits")
    >>> qm = QualityMetrics(lc, binsize=1.0, emin=20, emax=100)
    >>>
    >>> # Compute various chi-squared metrics
    >>> raw_chi = qm.raw_chi_squared()
    >>> clip_chi = qm.sigma_clip_chi_squared(sigma=3.0)
    >>> gti_chi = qm.gti_chi_squared()
    >>>
    >>> print(f"Raw: {raw_chi:.2f}, Clipped: {clip_chi:.2f}, GTI: {gti_chi:.2f}")

    See Also
    --------
    raw_chi_squared : Basic chi-squared test
    sigma_clip_chi_squared : Remove outliers before testing
    gti_chi_squared : Test only good time intervals
    """

    def __init__(
        self,
        lightcurve: Optional[LightCurve] = None,
        binsize: float = 1.0,
        emin: float = 1.0,
        emax: float = 1000.0,
        local_time: bool = False,
    ) -> None:
        """Initialize QualityMetrics instance."""
        if lightcurve is not None and not isinstance(lightcurve, LightCurve):
            raise TypeError(f"lightcurve must be LightCurve instance or None, got {type(lightcurve)}")

        if binsize <= 0:
            raise ValueError(f"binsize must be positive, got {binsize}")

        if emin >= emax:
            raise ValueError(f"emin ({emin}) must be less than emax ({emax})")

        if emin < 0:
            raise ValueError(f"emin must be non-negative, got {emin}")

        self.lightcurve = lightcurve
        self.binsize = binsize
        self.emin = emin
        self.emax = emax
        self.local_time = local_time
        self.module_data: Optional[dict] = None

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"QualityMetrics(binsize={self.binsize}s, "
            f"energy=({self.emin:.1f}-{self.emax:.1f}) keV, "
            f"lightcurve={'set' if self.lightcurve else 'None'})"
        )

    def _compute_counts(self) -> dict:
        """
        Compute or retrieve cached rebinned counts for all modules.

        Returns
        -------
        dict
            Dictionary with 'time' (ndarray) and 'counts' (ndarray, shape (8, n_bins))

        Raises
        ------
        ValueError
            If lightcurve is not set
        """
        if self.lightcurve is None:
            raise ValueError("Lightcurve must be set before computing counts")

        if self.module_data is not None:
            return self.module_data

        time, counts = self.lightcurve.rebin_by_modules(
            binsize=self.binsize,
            emin=self.emin,
            emax=self.emax,
            local_time=self.local_time,
        )

        self.module_data = {
            "time": time,
            "counts": np.asarray(counts),  # Shape: (8, n_bins)
        }

        return self.module_data

    def _compute_chi_squared_red(
        self,
        counts: NDArray[np.float64],
        return_all: bool = False,
    ) -> Union[float, Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]]:
        """
        Compute reduced chi-squared for count data.

        Parameters
        ----------
        counts : ndarray
            Count array(s). Shape: (n_modules, n_bins) or (n_bins,)
        return_all : bool, default False
            If True, return (chi_squared, dof, total_counts) per module.
            If False, return weighted mean chi-squared.

        Returns
        -------
        chi_squared_red : float
            Weighted mean of chisq/dof across modules (if return_all=False)
        chi_squared, dof, total_counts : tuple of ndarrays
            Per-module statistics (if return_all=True)

        Notes
        -----
        - Empty bins (counts=0) are treated as NaN and excluded
        - DOF = (number of non-empty bins) - 1
        - Weighting by total counts gives more influence to active modules
        """
        counts = np.asarray(counts)

        # Replace zeros with NaN (exclude empty bins)
        counts = np.where(counts == 0, np.nan, counts)

        # Compute mean and chi-squared per module
        mean_counts = np.nanmean(counts, axis=-1, keepdims=True)
        chi_squared = np.nansum((counts - mean_counts) ** 2 / mean_counts, axis=-1)

        # DOF = number of non-empty bins minus 1
        nan_mask = ~np.isnan(counts)
        dof = np.sum(nan_mask, axis=-1) - 1
        total_counts = np.nansum(counts, axis=-1)

        if return_all:
            return chi_squared, dof, total_counts

        # Return weighted mean
        if np.sum(total_counts) == 0 or np.all(dof <= 0):
            return np.nan

        # Weight by total counts (more counts = more reliable chi-squared)
        valid_mask = dof > 0
        chi_squared_red = chi_squared[valid_mask] / dof[valid_mask]
        weights = total_counts[valid_mask]

        return np.average(chi_squared_red, weights=weights)

    def raw_chi_squared(
        self,
        counts: Optional[NDArray[np.float64]] = None,
        return_all: bool = False,
    ) -> Union[float, Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]]:
        """
        Compute raw reduced chi-squared (no filtering).

        Tests whether count rates are consistent with Poisson statistics.
        Values near 1.0 indicate stable, constant background.

        Parameters
        ----------
        counts : ndarray, optional
            Count array(s) to analyze. If None, uses cached module data.
        return_all : bool, default False
            If True, return per-module results. If False, return weighted mean.

        Returns
        -------
        chi_squared_red : float
            Reduced chi-squared (chisq/dof)

        Examples
        --------
        >>> qm = QualityMetrics(lc, binsize=1.0, emin=20, emax=100)
        >>> chi = qm.raw_chi_squared()
        >>> print(f"chisq/dof = {chi:.2f}")

        >>> # Get per-module results
        >>> chi_vals, dof, counts = qm.raw_chi_squared(return_all=True)
        >>> for i, (c, d) in enumerate(zip(chi_vals, dof)):
        ...     print(f"Module {i}: chisq = {c:.1f}, dof = {d}")
        """
        if counts is None:
            counts = self._compute_counts()["counts"]

        return self._compute_chi_squared_red(counts, return_all=return_all)

    def sigma_clip_chi_squared(
        self,
        sigma: float = 3.0,
        counts: Optional[NDArray[np.float64]] = None,
        return_all: bool = False,
    ) -> Union[float, Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]]:
        """
        Compute sigma-clipped reduced chi-squared.

        Removes outlier bins (>sigma standard deviations from mean)
        before computing chi-squared. Useful for detecting transient
        flares or background instabilities.

        Parameters
        ----------
        sigma : float, default 3.0
            Sigma clipping threshold in standard deviations
        counts : ndarray, optional
            Count array(s) to analyze. If None, uses cached module data.
        return_all : bool, default False
            If True, return per-module results.

        Returns
        -------
        chi_squared_red : float
            Reduced chi-squared after clipping outliers

        Examples
        --------
        >>> qm = QualityMetrics(lc, binsize=1.0, emin=20, emax=100)
        >>>
        >>> # Conservative clipping (remove extreme outliers)
        >>> chi_3sig = qm.sigma_clip_chi_squared(sigma=3.0)
        >>>
        >>> # Aggressive clipping (remove moderate outliers)
        >>> chi_1sig = qm.sigma_clip_chi_squared(sigma=1.0)
        >>>
        >>> print(f"3sigma: {chi_3sig:.2f}, 1sigma: {chi_1sig:.2f}")

        Notes
        -----
        Lower chi-squared after clipping indicates presence of outliers
        (flares, background jumps, etc.)
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        if counts is None:
            counts = self._compute_counts()["counts"]

        # Compute mean and std per module
        mean_count = np.nanmean(counts, axis=-1, keepdims=True)
        std_count = np.nanstd(counts, axis=-1, keepdims=True)

        # Mask outliers
        mask = np.abs(counts - mean_count) < sigma * std_count
        filtered_counts = np.where(mask, counts, np.nan)

        return self._compute_chi_squared_red(filtered_counts, return_all=return_all)

    def gti_chi_squared(
        self,
        time: Optional[NDArray[np.float64]] = None,
        counts: Optional[NDArray[np.float64]] = None,
        gtis: Optional[NDArray[np.float64]] = None,
        return_all: bool = False,
    ) -> Union[float, Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]]:
        """
        Compute GTI-filtered reduced chi-squared.

        Only uses bins within Good Time Intervals (GTIs).
        Useful for excluding known bad data periods.

        Parameters
        ----------
        time : ndarray, optional
            Time array. If None, uses cached module data.
        counts : ndarray, optional
            Count array(s). If None, uses cached module data.
        gtis : ndarray, optional
            Good Time Intervals (N, 2) array in IJD.
            If None, uses lightcurve.gtis.
        return_all : bool, default False
            If True, return per-module results.

        Returns
        -------
        chi_squared_red : float
            Reduced chi-squared within GTIs only

        Raises
        ------
        ValueError
            If no overlap between GTIs and time range

        Examples
        --------
        >>> qm = QualityMetrics(lc, binsize=1.0, emin=20, emax=100)
        >>> chi_gti = qm.gti_chi_squared()
        >>> print(f"GTI-filtered chisq/dof = {chi_gti:.2f}")
        >>>
        >>> # Use custom GTIs
        >>> custom_gtis = np.array([[3000.0, 3100.0], [3200.0, 3300.0]])
        >>> chi_custom = qm.gti_chi_squared(gtis=custom_gtis)

        Notes
        -----
        GTIs are always in IJD format, regardless of local_time setting.
        Time array must be converted to IJD for comparison.
        """
        if counts is None or time is None:
            data = self._compute_counts()
            time, counts = data["time"], data["counts"]

        if gtis is None:
            if self.lightcurve is None:
                raise ValueError("Must provide gtis or set lightcurve")
            gtis = self.lightcurve.gtis

        # Check for overlap
        if gtis[0, 0] > time[-1] or gtis[-1, 1] < time[0]:
            raise ValueError(
                f"No overlap between GTIs ({gtis[0,0]:.1f}-{gtis[-1,1]:.1f}) "
                f"and time range ({time[0]:.1f}-{time[-1]:.1f}). "
                "Verify time is in IJD format."
            )

        # Create GTI mask
        gti_mask = np.zeros_like(time, dtype=bool)
        for gti_start, gti_stop in gtis:
            gti_mask |= (time >= gti_start) & (time <= gti_stop)

        # Apply mask (set non-GTI bins to NaN)
        filtered_counts = np.where(gti_mask, counts, np.nan)

        return self._compute_chi_squared_red(filtered_counts, return_all=return_all)
