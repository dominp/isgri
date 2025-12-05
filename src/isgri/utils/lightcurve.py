"""
ISGRI Light Curve Analysis
===========================

Tools for working with INTEGRAL/ISGRI event data and creating light curves.

Classes
-------
LightCurve : Main light curve class

Examples
--------
>>> from isgri.utils import LightCurve
>>>
>>> # Load events with PIF weighting
>>> lc = LightCurve.load_data(
...     events_path="events.fits",
...     pif_path="model.fits",
...     pif_threshold=0.5
... )
>>>
>>> # Create 1-second binned light curve
>>> time, counts = lc.rebin(binsize=1.0, emin=20, emax=100)
>>>
>>> # Analyze by detector module
>>> times, module_lcs = lc.rebin_by_modules(1.0, 20, 100)
"""

from astropy.io import fits
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, Tuple, List
from pathlib import Path
import os
from .file_loaders import load_isgri_events, load_isgri_pif, default_pif_metadata, merge_metadata
from .pif import DETZ_BOUNDS, DETY_BOUNDS


class LightCurve:
    """
    ISGRI light curve analysis class.

    Handles event data with optional detector response (PIF) weighting.
    Provides rebinning, module-level analysis, and time conversions.

    Parameters
    ----------
    time : ndarray
        IJD time values of events
    energies : ndarray
        Energy values in keV
    gtis : ndarray
        Good Time Intervals (start, stop) in IJD
    dety : ndarray
        Y detector coordinates (mm)
    detz : ndarray
        Z detector coordinates (mm)
    weights : ndarray
        PIF weight for each event (1.0 if no PIF)
    metadata : dict
        Event metadata (SWID, source info, etc.)

    Attributes
    ----------
    t0 : float
        Reference time (first event time in IJD)
    local_time : ndarray
        Time relative to t0 in seconds

    Examples
    --------
    >>> lc = LightCurve.load_data("events.fits", pif_path="model.fits")
    >>> time, counts = lc.rebin(binsize=1.0, emin=20, emax=100)
    >>> print(f"Total counts: {counts.sum()}")

    >>> # Module-by-module analysis
    >>> times, module_counts = lc.rebin_by_modules(1.0, 20, 100)
    >>> print(f"Module 3 counts: {module_counts[3].sum()}")

    See Also
    --------
    load_data : Load events from FITS files
    rebin : Rebin light curve
    rebin_by_modules : Rebin by detector module
    """

    time: NDArray[np.float64]
    energies: NDArray[np.float64]
    gtis: NDArray[np.float64]
    t0: float
    local_time: NDArray[np.float64]
    dety: NDArray[np.float64]
    detz: NDArray[np.float64]
    weights: NDArray[np.float64]
    metadata: dict

    def __init__(
        self,
        time: NDArray[np.float64],
        energies: NDArray[np.float64],
        gtis: NDArray[np.float64],
        dety: NDArray[np.float64],
        detz: NDArray[np.float64],
        weights: NDArray[np.float64],
        metadata: dict,
    ) -> None:
        """
        Initialize LightCurve instance.

        Parameters
        ----------
        time : ndarray
            IJD time values
        energies : ndarray
            Energy values in keV
        gtis : ndarray
            Good Time Intervals
        dety : ndarray
            Y detector coordinates
        detz : ndarray
            Z detector coordinates
        weights : ndarray
            PIF weights for each event
        metadata : dict
            Event metadata
        """
        self.time = time
        self.energies = energies
        self.gtis = gtis
        self.t0 = time[0]
        self.local_time = (time - self.t0) * 86400
        self.dety = dety
        self.detz = detz
        self.weights = weights
        self.metadata = metadata

    @classmethod
    def load_data(
        cls,
        events_path: Optional[Union[str, Path]] = None,
        pif_path: Optional[Union[str, Path]] = None,
        scw: Optional[str] = None,
        source: Optional[str] = None,
        pif_threshold: float = 0.5,
        pif_extension: int = -1,
    ) -> "LightCurve":
        """
        Loads the events from the given events file and PIF file (optional).

        Args:
            events_path (str): The path to the events file or directory.
            pif_path (str, optional): The path to the PIF file. Defaults to None.
            scw (str, optional): SCW identifier for auto-path resolution. Defaults to None.
            source (str, optional): Source name for auto-path resolution. Defaults to None.
            pif_threshold (float, optional): The PIF threshold value. Defaults to 0.5.
            pif_extension (int, optional): PIF file extension index. Defaults to -1.

        Returns:
            LightCurve: An instance of the LightCurve class.
        """
        events, gtis, metadata = load_isgri_events(events_path)
        if pif_path:
            if pif_threshold < 0 or pif_threshold > 1:
                raise ValueError(f"pif_threshold must be in [0, 1], got {pif_threshold}")
            
            events, weights, metadata_pif = load_isgri_pif(pif_path, events, pif_threshold, pif_extension)
        else:
            weights = np.ones(len(events))
            metadata_pif = default_pif_metadata()

        metadata = merge_metadata(metadata, metadata_pif)
        time = events["TIME"]
        energies = events["ISGRI_ENERGY"]
        dety, detz = events["DETY"], events["DETZ"]
        return cls(time, energies, gtis, dety, detz, weights, metadata)

    def rebin(
        self,
        binsize: Union[float, NDArray[np.float64], List[float]],
        emin: float,
        emax: float,
        local_time: bool = True,
        custom_mask: Optional[NDArray[np.bool_]] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Rebins the events with the specified bin size and energy range.

        Args:
            binsize (float or array): The bin size in seconds, or array of bin edges.
            emin (float): The minimum energy value in keV.
            emax (float): The maximum energy value in keV.
            local_time (bool, optional): If True, returns local time. If False, returns IJD time. Defaults to True.
            custom_mask (ndarray, optional): Additional boolean mask to apply. Defaults to None.

        Returns:
            tuple: (time, counts) arrays.

        Raises:
            ValueError: If emin >= emax.

        Examples:
            >>> time, counts = lc.rebin(binsize=1.0, emin=30, emax=300)
            >>> time, counts = lc.rebin(binsize=[0, 1, 2, 5, 10], emin=50, emax=200)
        """
        # Validate inputs
        if emin >= emax:
            raise ValueError(f"emin ({emin}) must be less than emax ({emax})")

        if emin < 0:
            raise ValueError(f"emin must be non-negative, got {emin}")

        if isinstance(binsize, (int, float)) and binsize <= 0:
            raise ValueError(f"binsize must be positive, got {binsize}")

        if custom_mask is not None and len(custom_mask) != len(self.time):
            raise ValueError(
                f"custom_mask length ({len(custom_mask)}) must match " f"number of events ({len(self.time)})"
            )

        # Select time axis
        time = self.local_time if local_time else self.time
        t0 = 0 if local_time else self.t0

        # Create bins
        bins, binsize_actual = self._create_bins(binsize, time, t0, local_time)

        # Apply filters
        mask = self._create_event_mask(emin, emax, custom_mask)
        time_filtered = time[mask]
        weights_filtered = self.weights[mask]

        # Histogram
        counts, bin_edges = np.histogram(time_filtered, bins=bins, weights=weights_filtered)
        time_centers = bin_edges[:-1] + 0.5 * binsize_actual

        return time_centers, counts

    def _create_bins(
        self,
        binsize: Union[float, NDArray[np.float64], List[float]],
        time: NDArray[np.float64],
        t0: float,
        local_time: bool,
    ) -> Tuple[NDArray[np.float64], float]:
        """
        Create time bins for rebinning.

        Args:
            binsize (float or array): Bin size or custom bin edges.
            time (ndarray): Time array.
            t0 (float): Start time.
            local_time (bool): Whether using local time.

        Returns:
            tuple: (bins array, actual binsize).
        """
        if isinstance(binsize, (list, np.ndarray)):
            # Custom bin edges provided
            bins = np.array(binsize)
            binsize_actual = np.mean(np.diff(bins))
        else:
            # Uniform binning
            binsize_actual = binsize if local_time else binsize / 86400
            bins = np.arange(t0, time[-1] + binsize_actual, binsize_actual)

        return bins, binsize_actual

    def _create_event_mask(
        self,
        emin: float,
        emax: float,
        custom_mask: Optional[NDArray[np.bool_]] = None,
    ) -> NDArray[np.bool_]:
        """
        Create combined event filter mask.

        Args:
            emin (float): Minimum energy in keV.
            emax (float): Maximum energy in keV.
            custom_mask (ndarray, optional): Additional mask to apply.

        Returns:
            ndarray: Boolean mask for events.
        """
        # Energy filter
        mask = (self.energies >= emin) & (self.energies < emax)

        # Custom filter (optional)
        if custom_mask is not None:
            mask &= custom_mask

        return mask

    def rebin_by_modules(
        self,
        binsize: float,
        emin: float,
        emax: float,
        local_time: bool = True,
        custom_mask: Optional[NDArray[np.bool_]] = None,
    ) -> Tuple[NDArray[np.float64], List[NDArray[np.float64]]]:
        """
        Rebins the events by all 8 detector modules with the specified bin size and energy range.

        Args:
            binsize (float): The bin size in seconds.
            emin (float): The minimum energy value in keV.
            emax (float): The maximum energy value in keV.
            local_time (bool, optional): If True, returns local time. Defaults to True.
            custom_mask (ndarray, optional): A custom mask to apply. Defaults to None.

        Returns:
            tuple: (times, counts) where:
                - times: array of time bin centers
                - counts: list of 8 arrays, one for each module

        Raises:
            ValueError: If emin >= emax.

        Examples:
            >>> times, counts = lc.rebin_by_modules(binsize=1.0, emin=30, emax=300)
            >>> module_3_lc = counts[3]  # Get lightcurve for module 3
        """
        if emin >= emax:
            raise ValueError("emin must be less than emax")

        time = self.local_time if local_time else self.time
        t0 = 0 if local_time else self.t0
        binsize_adj = binsize if local_time else binsize / 86400
        bins = np.arange(t0, time[-1] + binsize_adj, binsize_adj)
        times = bins[:-1] + 0.5 * binsize_adj

        energy_mask = (self.energies >= emin) & (self.energies < emax)
        if custom_mask is not None:
            energy_mask &= custom_mask

        time_filtered = time[energy_mask]
        dety_filtered = self.dety[energy_mask]
        detz_filtered = self.detz[energy_mask]
        weights_filtered = self.weights[energy_mask]

        # Compute module indices using digitize
        dety_bin = np.digitize(dety_filtered, DETY_BOUNDS) - 1  # 0 or 1
        detz_bin = np.digitize(detz_filtered, DETZ_BOUNDS) - 1  # 0, 1, 2, or 3
        module_idx = dety_bin + detz_bin * 2  # Flat index: 0-7

        counts = []
        for i in range(8):
            mask = module_idx == i
            counts.append(np.histogram(time_filtered[mask], bins=bins, weights=weights_filtered[mask])[0])

        return times, counts

    def cts(
        self,
        t1: float,
        t2: float,
        emin: float,
        emax: float,
        local_time: bool = True,
    ) -> float:
        """
        Calculates the counts in the specified time and energy range.

        Args:
            t1 (float): The start time (seconds or IJD depending on local_time).
            t2 (float): The end time (seconds or IJD depending on local_time).
            emin (float): The minimum energy value in keV.
            emax (float): The maximum energy value in keV.
            local_time (bool, optional): If True, uses local time. Defaults to True.

        Returns:
            float: The total counts in the specified range.
        """
        time = self.local_time if local_time else self.time
        return np.sum(self.weights[(time >= t1) & (time < t2) & (self.energies >= emin) & (self.energies < emax)])

    def ijd2loc(self, ijd_time: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Converts IJD (INTEGRAL Julian Date) time to local time.

        Args:
            ijd_time (float or ndarray): The IJD time value(s).

        Returns:
            float or ndarray: The local time in seconds from t0.
        """
        return (ijd_time - self.t0) * 86400

    def loc2ijd(self, evt_time: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Converts local time to IJD (INTEGRAL Julian Date) time.

        Args:
            evt_time (float or ndarray): The local time in seconds from t0.

        Returns:
            float or ndarray: The IJD time value(s).
        """
        return evt_time / 86400 + self.t0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LightCurve(n_events={len(self.time)}, "
            f"time_range=({self.time[0]:.3f}, {self.time[-1]:.3f}) IJD, "
            f"energy_range=({self.energies.min():.1f}, {self.energies.max():.1f}) keV, "
            f"scw={self.metadata.get('SWID', 'Unknown')})"
        )
