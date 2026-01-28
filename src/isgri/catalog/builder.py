"""ISGRI Catalog Builder
======================

Tools for building and updating INTEGRAL/ISGRI science window catalogs.

Classes
-------
CatalogBuilder : Main catalog builder class

Examples
--------
>>> from isgri.catalog.builder import CatalogBuilder
>>>
>>> # Create builder instance
>>> builder = CatalogBuilder(
...     archive_path="/path/to/archive",
...     catalog_path="/path/to/catalog.fits",
...     lightcurve_cache="/path/to/cache",
...     n_cores=8
... )
>>>
>>> # Update catalog with new science windows
>>> builder.update_catalog()
"""

from isgri.utils import LightCurve, QualityMetrics
from ..config import Config
import numpy as np
import os, subprocess, glob
from typing import Optional
from joblib import Parallel, delayed  # type: ignore
import multiprocessing
from collections import defaultdict
from astropy.table import Table, vstack
from pathlib import Path

new_catalog_names = [
    "REVOL",
    "SWID",
    "TSTART",
    "ONTIME",
    "TSTOP",
    "RA_SCX",
    "DEC_SCX",
    "RA_SCZ",
    "DEC_SCZ",
    "NoEVTS",
    "CHI",
    "CUT_CHI",
    "GTI_CHI",
]
new_catalog_dtypes = ["i8", "S12", "f8", "f8", "f8", "f8", "f8", "f8", "f8", "i8", "f8", "f8", "f8"]


class CatalogBuilder:
    """ISGRI catalog builder and updater.

    Processes INTEGRAL/ISGRI science windows to build catalogs containing
    quality metrics, pointing information, and light curve data.

    Parameters
    ----------
    archive_path : str, optional
        Path to INTEGRAL archive directory. If None, uses config file.
    catalog_path : str, optional
        Path to catalog FITS file. If None, uses config file.
    lightcurve_cache : str, optional
        Path to directory for caching light curve arrays. If None, no caching.
    n_cores : int, optional
        Number of CPU cores for parallel processing. If None, uses all available cores.

    Attributes
    ----------
    archive_path : str
        Path to INTEGRAL archive
    catalog_path : str
        Path to catalog file
    lightcurve_cache : str or None
        Path to light curve cache directory
    n_cores : int
        Number of parallel workers
    catalog : astropy.table.Table
        Loaded catalog table

    Examples
    --------
    >>> builder = CatalogBuilder(
    ...     archive_path="/data/integral",
    ...     catalog_path="catalog.fits",
    ...     n_cores=4
    ... )

    >>> # Update catalog with new observations
    >>> builder.update_catalog()

    >>> # Find all science windows
    >>> swids, paths = builder.find_scws()
    >>> print(f"Found {len(swids)} science windows")

    See Also
    --------
    ScwQuery : Query and filter catalog data
    LightCurve : Light curve analysis
    QualityMetrics : Quality metric computation
    """

    def __init__(
        self,
        archive_path: Optional[str] = None,
        catalog_path: Optional[str] = None,
        lightcurve_cache: Optional[str] = None,
        n_cores: Optional[int] = None,
    ):
        if archive_path is None or catalog_path is None:
            cfg = Config()
            if archive_path is None:
                archive_path = cfg.archive_path
            if catalog_path is None:
                catalog_path = cfg.catalog_path
        if catalog_path is None:
            raise FileNotFoundError("Catalog path must be specified either in arguments or config file.")
        self.archive_path = archive_path
        self.catalog_path = catalog_path
        self.lightcurve_cache = lightcurve_cache
        self.n_cores = n_cores if n_cores is not None else multiprocessing.cpu_count()
        self.catalog = self._load_catalog()

    def _load_catalog(self):
        """Load existing catalog or create new empty catalog.

        Returns
        -------
        astropy.table.Table
            Loaded catalog table or new empty table.

        Raises
        ------
        FileNotFoundError
            If catalog directory does not exist.
        """
        catalog_path = Path(self.catalog_path)
        if catalog_path.is_file():
            return Table.read(catalog_path)
        elif catalog_path.parent.is_dir():
            print("Catalog file not found, creating new catalog.")
            return Table(names=new_catalog_names, dtype=new_catalog_dtypes)
        else:
            raise FileNotFoundError(f"Directory for catalog does not exist: {catalog_path.parent}")

    def _add_catalog_data(self, table_data_rows: list[dict]):
        """Add new rows to catalog and save to disk.

        Parameters
        ----------
        table_data_rows : list of dict
            List of dictionaries containing catalog row data.
            Each dict must have keys matching catalog column names.

        Notes
        -----
        Updates are written atomically using a temporary file to prevent corruption.
        The catalog is sorted by TSTART after adding new data.
        """
        new_data = Table(rows=table_data_rows, names=new_catalog_names, dtype=new_catalog_dtypes)
        self.catalog = vstack([self.catalog, new_data])
        self.catalog.sort("TSTART")

        temp_catalog_path = Path(self.catalog_path).with_suffix(".tmp")
        self.catalog.write(temp_catalog_path, overwrite=True, format="fits")
        os.replace(temp_catalog_path, self.catalog_path)

    def _add_array_data(self, rev: str, array_data: np.ndarray):
        """Add light curve array data to cache for a revolution.

        Parameters
        ----------
        rev : str
            Revolution number (4-digit string, e.g., '0011').
        array_data : ndarray
            Structured array containing SWID, TIME, COUNTS, MODULE_COUNTS, and GTIS.

        Raises
        ------
        ValueError
            If lightcurve_cache path is not set.

        Notes
        -----
        Merges new data with existing revolution data if present.
        Saves as NumPy .npy file named by revolution number.
        """
        if self.lightcurve_cache is None:
            raise ValueError("Lightcurve cache path is not set.")
        file_path = Path(self.lightcurve_cache) / f"{int(rev):0>4}.npy"

        if file_path.exists():
            old_data = np.load(file_path, allow_pickle=True)
            mask = ~np.isin(old_data["SWID"], array_data["SWID"])
            array_data = np.concatenate([old_data[mask], array_data])

        np.save(file_path, array_data)

    def _process_scw(self, path) -> tuple[dict, list]:
        """Process a single science window and compute quality metrics.

        Parameters
        ----------
        path : str
            Path to ISGRI events FITS file.

        Returns
        -------
        table_data : dict
            Catalog row data containing metadata and quality metrics.
        array_data : dict
            Light curve data (time, counts, modules, GTIs).

        Notes
        -----
        Computes three quality metrics:
        - CHI: Raw chi-squared
        - CUT_CHI: Sigma-clipped chi-squared
        - GTI_CHI: GTI-filtered chi-squared (NaN if insufficient GTI coverage)

        Light curves are binned at 1 second resolution in 15-1000 keV band.
        """
        lc = LightCurve.load_data(path)

        time, full_counts = lc.rebin(1, emin=15, emax=1000, local_time=False)
        _, module_counts = lc.rebin_by_modules(1, emin=15, emax=1000, local_time=False)
        module_counts.insert(0, full_counts)
        module_counts = np.array(module_counts)
        quality = QualityMetrics(lc)
        quality.module_data = {"time": time, "counts": module_counts[1:]}
        raw_chisq = quality.raw_chi_squared()
        clipped_chisq = quality.sigma_clip_chi_squared()
        try:
            gti_chisq = quality.gti_chi_squared()
        except ValueError:
            gti_chisq = np.nan

        # cnames = [
        #     ("REVOL", int),
        #     ("SWID", "S12"),
        #     ("TSTART", float),
        #     ("TSTOP", float),
        #     ("TELAPSE", float),
        #     ("RA_SCX", float),
        #     ("DEC_SCX", float),
        #     ("RA_SCZ", float),
        #     ("DEC_SCZ", float),
        #     ("NoEVTS", int),
        #     ("LCs", np.ndarray),
        #     ("GTIs", np.ndarray),
        #     ("CHI", float),
        #     ("CUT_CHI", float),
        #     ("GTI_CHI", float),
        # ]
        table_data = {
            "REVOL": lc.metadata["REVOL"],
            "SWID": lc.metadata["SWID"],
            "TSTART": lc.metadata["TSTART"],
            "TSTOP": lc.metadata["TSTOP"],
            "ONTIME": lc.metadata["TELAPSE"],
            "RA_SCX": lc.metadata["RA_SCX"],
            "DEC_SCX": lc.metadata["DEC_SCX"],
            "RA_SCZ": lc.metadata["RA_SCZ"],
            "DEC_SCZ": lc.metadata["DEC_SCZ"],
            "NoEVTS": len(lc.time),
            "CHI": raw_chisq,
            "CUT_CHI": clipped_chisq,
            "GTI_CHI": gti_chisq,
        }
        array_data = {
            "SWID": lc.metadata["SWID"],
            "TIME": time,
            "COUNTS": full_counts,
            "MODULE_COUNTS": module_counts[1:],
            "GTIS": lc.gtis,
        }
        return table_data, array_data

    def _process_rev(self, rev_paths: list[str]) -> tuple[list[dict], list[list]]:
        """Process all science windows in a revolution in parallel.

        Parameters
        ----------
        rev_paths : list of str
            Paths to event files for all ScWs in revolution.

        Returns
        -------
        table_data_list : list of dict
            Catalog rows for all processed ScWs.
        array_data : ndarray
            Structured array of light curve data for all ScWs.

        Notes
        -----
        Uses joblib for parallel processing across n_cores workers.
        """
        data = Parallel(n_jobs=self.n_cores, backend="multiprocessing")(
            delayed(self._process_scw)(path) for path in rev_paths
        )
        table_data_list, array_data_dicts = zip(*data)

        dtype = [("SWID", "U16"), ("TIME", "O"), ("COUNTS", "O"), ("MODULE_COUNTS", "O"), ("GTIS", "O")]
        array_data = np.empty(len(array_data_dicts), dtype=dtype)
        for i, d in enumerate(array_data_dicts):
            array_data[i] = (d["SWID"], d["TIME"], d["COUNTS"], d["MODULE_COUNTS"], d["GTIS"])
        return table_data_list, array_data

    def find_scws(self) -> tuple[np.ndarray[str], np.ndarray[str]]:
        """Find all science windows in the archive.

        Returns
        -------
        swids : ndarray of str
            Array of SWID identifiers (12 characters).
        swid_paths : ndarray of str
            Array of corresponding directory paths.

        Notes
        -----
        Only includes ScWs matching pattern with '0.0' (Pointings, slews are omitted) in directory name.
        Scans all revolution directories in archive_path.
        """
        # Find all SCW files in the archive
        revolutions = os.scandir(self.archive_path)
        swids, swid_paths = [], []
        for rev in revolutions:
            if not rev.is_dir():
                continue
            for scw in os.scandir(rev.path):
                swid = scw.name
                path = scw.path
                if len(swid) == 16 and "0.0" in swid:
                    swids.append(swid.split(".")[0])
                    swid_paths.append(path)
        return np.array(swids), np.array(swid_paths)

    def find_event_files(
        self, swids: np.ndarray[str], swid_paths: np.ndarray[str]
    ) -> tuple[np.ndarray[str], np.ndarray[str]]:
        """Filter science windows to those with event files.

        Parameters
        ----------
        swids : ndarray of str
            Array of SWID identifiers.
        swid_paths : ndarray of str
            Array of ScW directory paths.

        Returns
        -------
        valid_swids : ndarray of str
            SWIDs with existing event files.
        valid_paths : ndarray of str
            Paths to corresponding isgri_events.fits.gz files.

        Notes
        -----
        Checks for existence of 'isgri_events.fits.gz' in each ScW directory.
        """

        def check_file(swid, path):
            event_file = os.path.join(path, "isgri_events.fits.gz")
            return (swid, event_file) if os.path.exists(event_file) else None

        print("Checking for event files...")
        results = Parallel(n_jobs=self.n_cores, backend="threading")(
            delayed(check_file)(swid, path) for swid, path in zip(swids, swid_paths)
        )

        valid_data = [r for r in results if r is not None]
        if valid_data:
            valid_swids, valid_paths = zip(*valid_data)
            return np.array(valid_swids), np.array(valid_paths)
        return np.array([]), np.array([])

    def update_catalog(self):
        """Update catalog with new science windows from archive.

        Scans archive for new ScWs not present in catalog, processes them
        in parallel by revolution, and adds results to catalog and cache.

        Notes
        -----
        Processing workflow:
        1. Find all ScWs in archive
        2. Identify new ScWs not in catalog
        3. Filter to ScWs with event files
        4. Process by revolution in parallel
        5. Add to catalog and light curve cache

        Only ScWs with isgri_events.fits.gz files are processed.
        Progress is printed for each revolution.

        Examples
        --------
        >>> builder = CatalogBuilder()
        >>> builder.update_catalog()
        """
        print("Looking for ScWs in archive...")
        scws_in_archive, scws_paths = self.find_scws()
        print(f"Found {len(scws_in_archive)} ScWs in archive.")
        scws_in_catalog = np.array(self.catalog["SWID"], dtype=str)
        mask = np.isin(scws_in_archive, scws_in_catalog, invert=True)
        new_scws = scws_in_archive[mask]
        new_paths = scws_paths[mask]
        print(f"Found {len(new_scws)} new ScWs not in catalog.")
        to_process_scws, to_process_paths = self.find_event_files(new_scws, new_paths)
        print(f"{len(to_process_scws)} ScWs have event files and will be processed.")
        if len(to_process_scws) == 0:
            print("Exiting.")
            return

        revolutions = defaultdict(list)
        for swid, path in zip(to_process_scws, to_process_paths):
            revolutions[swid[:4]].append(path)
        revolutions = dict(sorted(revolutions.items()))
        for revolution, rev_paths in revolutions.items():
            print(f"Processing revolution {revolution} with {len(rev_paths)} ScWs...")
            table_data_rows, array_data_list = self._process_rev(rev_paths)
            self._add_catalog_data(table_data_rows)
            if self.lightcurve_cache is not None:
                self._add_array_data(revolution, array_data_list)
