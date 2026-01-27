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
        catalog_path = Path(self.catalog_path)
        if catalog_path.is_file():
            return Table.read(catalog_path)
        elif catalog_path.parent.is_dir():
            print("Catalog file not found, creating new catalog.")
            return Table(names=new_catalog_names, dtype=new_catalog_dtypes)
        else:
            raise FileNotFoundError(f"Directory for catalog does not exist: {catalog_path.parent}")

    def _add_catalog_data(self, table_data_rows: list[dict]):
        new_data = Table(rows=table_data_rows, names=new_catalog_names, dtype=new_catalog_dtypes)
        self.catalog = vstack([self.catalog, new_data])
        self.catalog.sort("TSTART")

        temp_catalog_path = Path(self.catalog_path).with_suffix(".tmp")
        self.catalog.write(temp_catalog_path, overwrite=True, format="fits")
        os.replace(temp_catalog_path, self.catalog_path)

    def _add_array_data(self, rev: str, array_data: np.ndarray):
        if self.lightcurve_cache is None:
            raise ValueError("Lightcurve cache path is not set.")
        file_path = Path(self.lightcurve_cache) / f"{int(rev):0>4}.npy"

        if file_path.exists():
            old_data = np.load(file_path, allow_pickle=True)
            mask = ~np.isin(old_data["SWID"], array_data["SWID"])
            array_data = np.concatenate([old_data[mask], array_data])

        np.save(file_path, array_data)

    def _process_scw(self, path) -> tuple[dict, list]:
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
        print("Checking for event files...")
        for idx, (swid, path) in enumerate(zip(swids, swid_paths)):
            event_file = os.path.join(path, "isgri_events.fits.gz")
            if os.path.exists(event_file):
                valid_swids.append(swid)
                valid_paths.append(event_file)
            if (idx + 1) % 100 == 0:
                print(f"Checked {idx + 1}/{len(swids)} ScWs...", end="\r")
        return np.array(valid_swids), np.array(valid_paths)

    def update_catalog(self):
        print("Looking for ScWs in archive...")
        scws_in_archive, scws_paths = self.find_scws()
        print(f"Found {len(scws_in_archive)} ScWs in archive.")
        scws_in_catalog = self.catalog["SWID"]
        mask = np.isin(scws_in_archive, scws_in_catalog, invert=True)
        new_scws = scws_in_archive[mask]
        new_paths = scws_paths[mask]
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
