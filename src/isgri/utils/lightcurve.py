from astropy.io import fits
import numpy as np
import os
from .file_loaders import load_isgri_events, load_isgri_pif, default_pif_metadata, merge_metadata


class LightCurve:
    """
    A class for working with isgri events. Works fully with and without isgri model file (further pif file).

    Attributes:
        time (ndarray): The time values of the events.
        energies (ndarray): The energy values of the events.
        gtis (ndarray): The Good Time Intervals (GTIs) of the events.
        t0 (float): The fitst time of the events.
        local_time (ndarray): The local time values of the events.
        pif (ndarray): The PIF values of the events.
        pif_ra (float): The right ascension of the PIF source.
        pif_dec (float): The declination of the PIF source.
        pif_source (str): The name of the PIF source.
        pif_cod (float): The COD (Coding Fraction) value of the PIF source.

    Methods:
        load_data(cls, events_path, pif_path=None, pif_threshold=0.5):
            Loads the light curve data from the given events file and PIF file (optional). It assumes that the source is in last extension of the PIF file.
        rebin(self, binsize, emin=30, emax=300, local=True):
            Rebins the light curve with the specified bin size and energy range.
        cts(self, t1, t2, emin, emax, format="s"):
            Calculates the counts in the specified time and energy range.
        evt_time(self, ijd_time):
            Converts IJD time to local time (start at t=0 s).
        ijt_time(self, evt_time):
            Converts local time to IJD

    """

    def __init__(self, time, energies, gtis, dety, detz, pif, metadata):
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
        Loads the events from the given events file and PIF file (optional). It assumes that the source is in last extension of the PIF file.

        Args:
            events_path (str): The path to the events file.
            pif_path (str, optional): The path to the PIF file. Defaults to None.
            pif_threshold (float, optional): The PIF threshold value. Defaults to 0.5.

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

    def rebin(self, binsize, emin=30, emax=300, local_time=True, custom_mask=None):
        """
        Rebins the events with the specified bin size and energy range.

        Args:
            binsize (float): The bin size in milliseconds.
            emin (float, optional): The minimum energy value. Defaults to 30.
            emax (float, optional): The maximum energy value. Defaults to 300.
            local (bool, optional): If True, the rebinned time values are returned in local time.
                                   If False, the rebinned time values are returned in IJD time.
                                   Defaults to True.

        Returns:
            ndarray: The rebinned time values.
            ndarray: The rebinned counts.

        Raises:
            None

        Examples:
            # Rebin the events with a bin size of 100 milliseconds and energy range between 50 and 200.
            time, counts = rebin(100, emin=50, emax=200)

        """
        
        mask = (self.energies >= emin) & (self.energies < emax)
        time = self.load_data if local_time else self.time
        time = time[mask]
        pif = self.pif
        pif = pif[mask]
        if custom_mask:
            time = time[custom_mask]
            pif = pif[custom_mask]

        binsize = (0.001 * binsize) / 86400
        bins = np.arange(self.t0, time[-1] + binsize, binsize)
        counts, histbins = np.histogram(time, bins=bins, weights=pif)
        time = np.array(histbins[:-1] + 0.5 * binsize)
        return time, counts

    def cts(self, t1, t2, emin, emax, format="s", bkg=False):
        """
        Calculates the counts in the specified time and energy range.

        Args:
            t1 (float): The start time.
            t2 (float): The end time.
            emin (float): The minimum energy value.
            emax (float): The maximum energy value.
            format (str, optional): The format of the time values. Defaults to "s".

        Returns:
            float: The total counts.

        """
        if format == "s":
            time = self.local_time
        else:
            time = self.time
        return np.sum(self.pif[(time > t1) & (time < t2) & (self.energies > emin) & (self.energies < emax)])

    def ijd2loc(self, ijd_time):
        """
        Converts IJD (International Julian Date) time to local time.

        Args:
            ijd_time (float): The IJD time.

        Returns:
            float: The event time.

        """
        return (ijd_time - self.t0) * 86400

    def loc2ijd(self, evt_time):
        """
        Converts event time to IJD.

        Args:
            evt_time (float): The event time.

        Returns:
            float: The IJD time.

        """
        return evt_time / 86400 + self.t0
