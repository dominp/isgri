from isgri.utils import LightCurve, QualityMetrics
from ..config import Config
import numpy as np
import os, subprocess, glob
from typing import Optional
from joblib import Parallel, delayed  # type: ignore
import multiprocessing
from collections import defaultdict
from astropy.table import Table


class CatalogBuilder:
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

        self.archive_path = archive_path
        self.catalog_path = catalog_path
        self.lightcurve_cache = lightcurve_cache
        self.n_cores = n_cores if n_cores is not None else multiprocessing.cpu_count()
        self.catalog = self._load_catalog()

    def _load_catalog(self):
        catalog = Table.read(self.catalog_path)
        return catalog

    def _process_scw(self, path) -> tuple[dict, list]:
        lc = LightCurve.load_data(path)

        time, full_counts = lc.rebin(1, emin=15, emax=1000, local_time=False)
        _, module_counts = lc.rebin_by_modules(1, emin=15, emax=1000, local_time=False)
        module_counts.insert(0, full_counts)
        module_counts = np.array(module_counts)
        quality = QualityMetrics.compute(lc)
        quality.module_data = {"time": time, "counts": module_counts[1:]}
        raw_chisq = quality.raw_chi_squared()
        clipped_chisq = quality.sigma_clip_chi_squared()
        gti_chisq = quality.gti_chi_squared()

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
        array_data = [lc.metadata["SWID"], time, module_counts, lc.gti]
        return table_data, array_data

    def _process_rev(self, rev_paths: list[str]) -> tuple[list[dict], list[list]]:
        data = Parallel(n_jobs=self.n_cores, backend="multiprocessing")(
            delayed(self._process_scw)(path) for path in rev_paths
        )
        table_data_list, array_data_list = zip(*data)
        return table_data_list, array_data_list

    def find_scws(self) -> tuple[np.ndarray[str], np.ndarray[str]]:
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
        valid_swids, valid_paths = [], []
        for swid, path in zip(swids, swid_paths):
            event_file = os.path.join(os.path.dirname(path), "isgri_events.fits.gz")
            if os.path.exists(event_file):
                valid_swids.append(swid)
                valid_paths.append(event_file)
        return np.array(valid_swids), np.array(valid_paths)

    def update_catalog(self):
        scws_in_archive, scws_paths = self.find_scws()
        scws_in_catalog = self.catalog["SWID"]
        mask = np.isin(scws_in_archive, scws_in_catalog, invert=True)
        to_process_scws = scws_in_archive[mask]
        to_process_paths = scws_paths[mask]
        if len(to_process_scws) == 0:
            print("Catalog is already up to date.")
            return

        revolutions = defaultdict(list)
        for swid, path in zip(to_process_scws, to_process_paths):
            revolutions[swid[:4]].append(path)

        for revolution, rev_paths in revolutions.items():
            print(f"Processing revolution {revolution} with {len(rev_paths)} SCWs...")
            table_data_rows, array_data_list = self._process_rev(rev_paths)
            self.catalog.add_entries(table_data_rows, array_data_list)
            print(f"Revolution {revolution} processed and catalog updated.")
