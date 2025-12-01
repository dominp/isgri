from email import header
from astropy.io import fits
import numpy as np
import os


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

        # if events_path is None:
        #     if scw is None:
        #         raise ValueError("Either events_path or scw must be provided.")
        #     events_path = f"{archive_path}/{scw[:4]}/{scw}.001/isgri_events.fits.gz"
        # if pif_path is None and source is not None:
        #     pif_path = f"{mask_path}/{source}/{scw[:4]}/{scw}_isgri_model.fits.gz"

        # if not os.path.exists(events_path):
        #     raise FileNotFoundError(f"Events file {events_path} not found.")
        # if pif_path and not os.path.exists(pif_path):
        #     raise FileNotFoundError(f"PIF file {pif_path} not found.")

        events, gtis, metadata = cls._load_events(cls, events_path)
        if pif_path is None:
            pif = np.ones(len(events))
            metadata_pif = {
                "SWID": metadata["SWID"],
                "SRC_RA": None,
                "SRC_DEC": None,
                "Source_Name": None,
                "cod": None,
                "No_of_Modules": 8,
            }

        else:
            events, pif, metadata_pif = cls._load_pif(pif_path, events, pif_threshold, pif_extension)
        if metadata["SWID"] != metadata_pif["SWID"]:
            raise ValueError("SWID mismatch between events and PIF files.")
        for key in metadata_pif:
            if key == "SWID":
                continue
            metadata[key] = metadata_pif[key]
        time = events["TIME"]
        energies = events["ISGRI_ENERGY"]
        dety, detz = events["DETY"], events["DETZ"]
        return cls(time, energies, gtis, dety, detz, pif, metadata)

    @staticmethod
    def _load_events(cls, events_path):
        with fits.open(events_path) as hdu:
            events = np.array(hdu["ISGR-EVTS-ALL"].data)
            header = hdu["ISGR-EVTS-ALL"].header
            metadata = {
                "REVOL": header.get("REVOL"),
                "SWID": header.get("SWID"),
                "TSTART": header.get("TSTART"),
                "TSTOP": header.get("TSTOP"),
                "TELAPSE": header.get("TELAPSE"),
                "OBT_TSTART": header.get("OBTSTART"),
                "OBT_TSTOP": header.get("OBTEND"),
                "RA_SCX": header.get("RA_SCX"),
                "DEC_SCX": header.get("DEC_SCX"),
                "RA_SCZ": header.get("RA_SCZ"),
                "DEC_SCZ": header.get("DEC_SCZ"),
            }
            try:
                gtis = np.array(hdu["IBIS-GNRL-GTI"].data)
                gtis = np.array([gtis["START"], gtis["STOP"]]).T
            except:
                gtis = np.array([events["TIME"][0], events["TIME"][-1]]).reshape(1, 2)
        events = events[events["SELECT_FLAG"] == 0]  # Filter out bad events
        return events, gtis, metadata

    @staticmethod
    def _load_pif(pif_path, events, pif_threshold=0.5, pif_extension=-1):
        with fits.open(pif_path) as hdu:
            pif_file = np.array(hdu[-1].data)
            header = hdu[-1].header
        metadata_pif = {
            "SWID": header.get("SWID"),
            "Source_ID": header.get("SOURCEID"),
            "Source_Name": header.get("NAME"),
            "SRC_RA": header.get("RA_OBJ"),
            "SRC_DEC": header.get("DEC_OBJ"),
            "No_of_Modules": LightCurve.__est_mods(pif_file),
            "cod": LightCurve._calc_cod(pif_file, events, pif_threshold),
        }
        pif_filter = pif_file > pif_threshold
        piffed_events = events[pif_filter[events["DETZ"], events["DETY"]]]
        pif = pif_file[piffed_events["DETZ"], piffed_events["DETY"]]
        return piffed_events, pif, metadata_pif

    @staticmethod
    def _calc_cod(pif_file, events, pif_threshold):
        pif_cod = pif_file == 1
        pif_cod = events[pif_cod[events["DETZ"], events["DETY"]]]
        cody = (np.max(pif_cod["DETY"]) - np.min(pif_cod["DETY"])) / 129
        codz = (np.max(pif_cod["DETZ"]) - np.min(pif_cod["DETZ"])) / 133
        pif_cod = codz * cody
        return pif_cod

    @staticmethod
    def __est_mods(mask):
        m, n = [0, 32, 66, 100, 134], [0, 64, 130]  # Separate modules
        mods = []
        for x1, x2 in zip(m[:-1], m[1:]):
            for y1, y2 in zip(n[:-1], n[1:]):
                a = mask[x1:x2, y1:y2].flatten()
                if len(a[a > 0.01]) / len(a) > 0.2:
                    mods.append(1)
                else:
                    mods.append(0)
        mods = np.array(mods)
        return mods

    def rebin(self, binsize, emin=30, emax=300, local=True, mask=None):
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
        binsize = (0.001 * binsize) / 86400
        if mask is not None:
            time = self.time[mask]
            pif = self.pif[mask]
            energies = self.energies[mask]
        else:
            time = self.time
            pif = self.pif
            energies = self.energies
        time = time[(energies >= emin) & (energies < emax)]
        pif = pif[(energies >= emin) & (energies < emax)]
        bins = np.arange(self.t0, time[-1] + binsize, binsize)
        counts, histbins = np.histogram(time, bins=bins, weights=pif)
        if local:
            time = np.array(((histbins[:-1] + 0.5 * binsize) - self.t0) * 86400)
        else:
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
