import numpy as np
from .lightcurve import LightCurve


class QualityMetrics:
    """
    A class for computing statistical quality metrics for ISGRI lightcurves.

    Attributes:
        lightcurve (LightCurve): The LightCurve instance to analyze.
        binsize (float): The bin size in seconds.
        emin (float): The minimum energy value in keV.
        emax (float): The maximum energy value in keV.
        local_time (bool): If True, uses local time. If False, uses IJD time. GTIs are always in IJD.
        module_data (dict): Cached rebinned data for all modules.

    Methods:
        raw_chi_squared: Computes raw reduced chi-squared.
        sigma_clip_chi_squared: Computes sigma-clipped reduced chi-squared.
        gti_chi_squared: Computes GTI-filtered reduced chi-squared.
    """

    def __init__(self, lightcurve: LightCurve | None = None, binsize=1.0, emin=1.0, emax=1000.0, local_time=False):
        """
        Initialize QualityMetrics instance.

        Args:
            lightcurve (LightCurve, optional): The LightCurve instance to analyze. Defaults to None.
            binsize (float, optional): The bin size in seconds. Defaults to 1.0.
            emin (float, optional): The minimum energy value in keV. Defaults to 1.0.
            emax (float, optional): The maximum energy value in keV. Defaults to 1000.0.
            local_time (bool, optional): If True, uses local time. If False, uses IJD time. Defaults to False.

        Raises:
            TypeError: If lightcurve is not a LightCurve instance or None.
        """
        if type(lightcurve) not in [LightCurve, type(None)]:
            raise TypeError("lightcurve must be an instance of LightCurve or None")
        self.lightcurve = lightcurve
        self.binsize = binsize
        self.emin = emin
        self.emax = emax
        self.local_time = local_time
        self.module_data = None

    def _compute_counts(self):
        """
        Compute or retrieve cached rebinned counts for all modules.

        Args:
            None
        Returns:
            dict: Dictionary with 'time' and 'counts' arrays.

        Raises:
            ValueError: If lightcurve is not set.
        """
        if self.lightcurve is None:
            raise ValueError("Lightcurve is not set.")
        if self.module_data is not None:
            return self.module_data
        time, counts = self.lightcurve.rebin_by_modules(
            binsize=self.binsize, emin=self.emin, emax=self.emax, local_time=self.local_time
        )
        module_data = {"time": time, "counts": counts}
        self.module_data = module_data
        return module_data

    def _compute_chi_squared_red(self, counts, return_all=False):
        """
        Compute reduced chi-squared for count data.

        Args:
            counts (ndarray): Count array(s) to analyze.
            return_all (bool, optional): If True, returns chi-squared for each array. If False, returns mean. Defaults to False.

        Returns:
            float or ndarray: Reduced chi-squared value(s).
        """
        counts = np.asarray(counts)
        counts = np.where(counts == 0, np.nan, counts)
        mean_counts = np.nanmean(counts, axis=-1, keepdims=True)
        chi_squared = np.nansum((counts - mean_counts) ** 2 / mean_counts, axis=-1)
        dof = counts.shape[-1] - 1
        if return_all:
            return chi_squared / dof
        return np.nanmean(chi_squared / dof)

    def raw_chi_squared(self, counts=None, return_all=False):
        """
        Computes raw reduced chi-squared for lightcurve data.

        Args:
            counts (ndarray, optional): Count array(s) to analyze. If None, uses cached module data. Defaults to None.
            return_all (bool, optional): If True, returns chi-squared for each module. If False, returns mean. Defaults to False.

        Returns:
            float or ndarray: Reduced chi-squared value(s).

        Examples:
            >>> qm = QualityMetrics(lc, binsize=1.0, emin=30, emax=300)
            >>> chi = qm.raw_chi_squared()
            >>> chi_all_modules = qm.raw_chi_squared(return_all=True)
        """
        if counts is None:
            counts = self._compute_counts()["counts"]
        return self._compute_chi_squared_red(counts, return_all=return_all)

    def sigma_clip_chi_squared(self, sigma=1.0, counts=None, return_all=False):
        """
        Computes sigma-clipped reduced chi-squared for lightcurve data.

        Args:
            sigma (float, optional): Sigma clipping threshold in standard deviations. Defaults to 1.0.
            counts (ndarray, optional): Count array(s) to analyze. If None, uses cached module data. Defaults to None.
            return_all (bool, optional): If True, returns chi-squared for each module. If False, returns mean. Defaults to False.

        Returns:
            float or ndarray: Reduced chi-squared value(s) after sigma clipping.

        Examples:
            >>> qm = QualityMetrics(lc, binsize=1.0, emin=30, emax=300)
            >>> chi = qm.sigma_clip_chi_squared(sigma=3.0)
        """
        if counts is None:
            counts = self._compute_counts()["counts"]
        mean_count = np.mean(counts, axis=-1, keepdims=True)
        std_count = np.std(counts, axis=-1, keepdims=True)
        mask = np.abs(counts - mean_count) < sigma * std_count
        filtered_counts = np.where(mask, counts, np.nan)
        return self._compute_chi_squared_red(filtered_counts, return_all=return_all)

    def gti_chi_squared(self, time=None, counts=None, gtis=None, return_all=False):
        """
        Computes GTI-filtered reduced chi-squared for lightcurve data.

        Args:
            time (ndarray, optional): Time array. If None, uses cached module data. Defaults to None.
            counts (ndarray, optional): Count array(s) to analyze. If None, uses cached module data. Defaults to None.
            gtis (ndarray, optional): Good Time Intervals (N, 2) array. If None, uses lightcurve GTIs. Defaults to None.
            return_all (bool, optional): If True, returns chi-squared for each module. If False, returns mean. Defaults to False.

        Returns:
            float or ndarray: Reduced chi-squared value(s) within GTIs only.

        Raises:
            ValueError: If no overlap between GTIs and lightcurve time range.

        Examples:
            >>> qm = QualityMetrics(lc, binsize=1.0, emin=30, emax=300)
            >>> chi = qm.gti_chi_squared()
        """
        if counts is None or time is None:
            data = self._compute_counts()
            time, counts = data["time"], data["counts"]
        if gtis is None:
            gtis = self.lightcurve.gtis
        if gtis[0, 0] > time[-1] or gtis[-1, 1] < time[0]:
            raise ValueError(
                "No overlap between GTIs and lightcurve time. If Lightcurve is set, verify time is in IJD."
            )
        gti_mask = np.zeros_like(time, dtype=bool)
        for gti_start, gti_stop in gtis:
            gti_mask |= (time >= gti_start) & (time <= gti_stop)
        filtered_counts = np.where(gti_mask, counts, np.nan)
        return self._compute_chi_squared_red(filtered_counts, return_all=return_all)
