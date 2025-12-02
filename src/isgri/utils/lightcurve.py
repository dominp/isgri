from astropy.io import fits
import numpy as np
import os
from .file_loaders import load_isgri_events, load_isgri_pif, default_pif_metadata, merge_metadata
from .pif import DETZ_BOUNDS, DETY_BOUNDS


class LightCurve:
    """
    A class for working with ISGRI events. Works fully with and without ISGRI model file (PIF file).

    Attributes:
        time (ndarray): The IJD time values of the events.
        energies (ndarray): The energy values of the events in keV.
        gtis (ndarray): The Good Time Intervals (GTIs) of the events.
        t0 (float): The first time of the events (IJD).
        local_time (ndarray): The local time values of the events (seconds from t0).
        dety (ndarray): The Y detector coordinates.
        detz (ndarray): The Z detector coordinates.
        pif (ndarray): The PIF values of the events.
        metadata (dict): Event metadata including SWID, source info, etc.

    Methods:
        load_data: Loads the light curve data from events and PIF files.
        rebin: Rebins the light curve with specified bin size and energy range.
        rebin_by_modules: Rebins the light curve for all 8 detector modules.
        cts: Calculates the counts in specified time and energy range.
        ijd2loc: Converts IJD time to local time (seconds from t0).
        loc2ijd: Converts local time to IJD time.
    """

    def __init__(self, time, energies, gtis, dety, detz, pif, metadata):
        """
        Initialize LightCurve instance.

        Args:
            time (ndarray): IJD time values.
            energies (ndarray): Energy values in keV.
            gtis (ndarray): Good Time Intervals.
            dety (ndarray): Y detector coordinates.
            detz (ndarray): Z detector coordinates.
            pif (ndarray): PIF mask values.
            metadata (dict): Event metadata.
        """
        self.time = time
        self.energies = energies
        self.gtis = gtis
        self.t0 = time[0]
        self.local_time = (time - self.t0) * 86400

        self.dety = dety
        self.detz = detz
        self.pif = pif
        self.metadata = metadata

    @classmethod
    def load_data(cls, events_path=None, pif_path=None, scw=None, source=None, pif_threshold=0.5, pif_extension=-1):
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
            events, pif, metadata_pif = load_isgri_pif(pif_path, events, pif_threshold, pif_extension)
        else:
            pif = np.ones(len(events))
            metadata_pif = default_pif_metadata()

        metadata = merge_metadata(metadata, metadata_pif)
        time = events["TIME"]
        energies = events["ISGRI_ENERGY"]
        dety, detz = events["DETY"], events["DETZ"]
        return cls(time, energies, gtis, dety, detz, pif, metadata)

    def rebin(self, binsize, emin, emax, local_time=True, custom_mask=None):
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
        if emin >= emax:
            raise ValueError("emin must be less than emax")

        # Select time axis
        time = self.local_time if local_time else self.time
        t0 = 0 if local_time else self.t0

        # Create bins
        bins, binsize_actual = self._create_bins(binsize, time, t0, local_time)

        # Apply filters
        mask = self._create_event_mask(emin, emax, custom_mask)
        time_filtered = time[mask]
        pif_filtered = self.pif[mask]

        # Histogram
        counts, bin_edges = np.histogram(time_filtered, bins=bins, weights=pif_filtered)
        time_centers = bin_edges[:-1] + 0.5 * binsize_actual

        return time_centers, counts

    def _create_bins(self, binsize, time, t0, local_time):
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

    def _create_event_mask(self, emin, emax, custom_mask=None):
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

    def rebin_by_modules(self, binsize, emin, emax, local_time=True, custom_mask=None):
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
        pif_filtered = self.pif[energy_mask]

        # Compute module indices using digitize
        dety_bin = np.digitize(dety_filtered, DETY_BOUNDS) - 1  # 0 or 1
        detz_bin = np.digitize(detz_filtered, DETZ_BOUNDS) - 1  # 0, 1, 2, or 3
        module_idx = dety_bin + detz_bin * 2  # Flat index: 0-7

        counts = []
        for i in range(8):
            mask = module_idx == i
            counts.append(np.histogram(time_filtered[mask], bins=bins, weights=pif_filtered[mask])[0])

        return times, counts

    def cts(self, t1, t2, emin, emax, local_time=True, bkg=False):
        """
        Calculates the counts in the specified time and energy range.

        Args:
            t1 (float): The start time (seconds or IJD depending on local_time).
            t2 (float): The end time (seconds or IJD depending on local_time).
            emin (float): The minimum energy value in keV.
            emax (float): The maximum energy value in keV.
            local_time (bool, optional): If True, uses local time. Defaults to True.
            bkg (bool, optional): Reserved for future background subtraction. Defaults to False.

        Returns:
            float: The total counts in the specified range.
        """
        time = self.local_time if local_time else self.time
        return np.sum(self.pif[(time >= t1) & (time < t2) & (self.energies >= emin) & (self.energies < emax)])

    def ijd2loc(self, ijd_time):
        """
        Converts IJD (INTEGRAL Julian Date) time to local time.

        Args:
            ijd_time (float or ndarray): The IJD time value(s).

        Returns:
            float or ndarray: The local time in seconds from t0.
        """
        return (ijd_time - self.t0) * 86400

    def loc2ijd(self, evt_time):
        """
        Converts local time to IJD (INTEGRAL Julian Date) time.

        Args:
            evt_time (float or ndarray): The local time in seconds from t0.

        Returns:
            float or ndarray: The IJD time value(s).
        """
        return evt_time / 86400 + self.t0
