from isgri.utils import LightCurve, QualityMetrics
import numpy as np
import os, subprocess
from .catalog_structure import CatalogStructure
from typing import Optional
from joblib import Parallel, delayed  # type: ignore
import multiprocessing


class CatalogBuilder:
    def __init__(
        self,
        archive_path: str,
        catalog_path: str,
        lightcurve_cache: Optional[str] = None,
        n_cores: Optional[int] = None,
    ):
        self.archive_path = archive_path
        self.catalog_path = catalog_path
        self.lightcurve_cache = lightcurve_cache
        self.n_cores = n_cores if n_cores is not None else multiprocessing.cpu_count()
        self.catalog = self._load_catalog()

    def _load_catalog(self):
        if not os.path.exists(self.catalog_path):
            empty_structure = CatalogStructure.get_empty_structure()
            return empty_structure
        else:
            catalog = CatalogStructure.load_from_fits(self.catalog_path)
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

    def _find_scws(self) -> tuple[np.ndarray[str], np.ndarray[str]]:
        # Find all SCW files in the archive
        scws_files = subprocess.run(
            ["find" , '-L', self.archive_path, "-name", "isgri_events.fits.gz"], capture_output=True, text=True
        )

