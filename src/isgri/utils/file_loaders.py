"""
ISGRI Data File Loaders
========================

Load INTEGRAL/ISGRI event files and detector response (PIF) files.

Functions
---------
load_isgri_events : Load photon events from FITS file
load_isgri_pif : Load and apply detector response (PIF) file
verify_events_path : Validate and resolve events file path
default_pif_metadata : Create default PIF metadata
merge_metadata : Combine events and PIF metadata

Examples
--------
>>> from isgri.utils import load_isgri_events, load_isgri_pif
>>>
>>> # Load events
>>> events, gtis, metadata = load_isgri_events("isgri_events.fits")
>>> print(f"Loaded {len(events)} events")
>>>
>>> # Apply PIF weighting
>>> filtered_events, pif_weights, pif_meta = load_isgri_pif(
...     "pif_model.fits",
...     events,
...     pif_threshold=0.5
... )
>>> print(f"Kept {len(filtered_events)}/{len(events)} events")
>>> print(f"Coding fraction: {pif_meta['cod']:.2%}")

Notes
-----
- Events files contain photon arrival times, energies, and detector positions
- PIF files contain detector response maps for specific source positions
- PIF weighting corrects for shadowing by the coded mask
"""

from astropy.io import fits
from astropy.table import Table
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Optional, Union
from pathlib import Path
import os
from .pif import coding_fraction, estimate_active_modules
from ..config import Config


def resolve_pif_path(
    pif_path: Optional[Union[str, Path]] = None,
    source: Optional[str] = None,
    swid: Optional[str] = None,
    config: Optional[Config] = None,
) -> str:
    """
    Resolve PIF file path based on input or source name.

    Parameters
    ----------
    pif_path : str or Path, optional
        Direct path to PIF file.
    source : str, optional
        Source name to construct default PIF path.
    swid : str, optional
        Science Window ID
    config : Config, optional
        Configuration object containing pif_path.

    Returns
    -------
    resolved_path : str
        Resolved absolute path to PIF file.

    Raises
    ------
    FileNotFoundError
        If PIF file not found.

    Notes
    -----
    - If both pif_path and source are provided, pif_path takes precedence.
    - If neither is provided, raises ValueError.
    """
    if pif_path is not None:
        path = Path(pif_path)
        if path.is_file():
            return str(path)
        if not path.is_dir():
            raise FileNotFoundError(f"PIF path is not a directory: {path}")

    elif config is not None and config.pif_path is not None:
        path = Path(config.pif_path)
        if not path.is_dir():
            raise FileNotFoundError(f"PIF path from config is not a directory: {path}")
    else:
        raise ValueError("Either pif_path must be provided here or in config.")

    if swid is None:
        raise ValueError("swid must be provided to resolve PIF file name.")

    if source is not None:
        candidate_path = path / source
        if candidate_path.is_dir():
            path = candidate_path

    revol = swid[:4]
    revol_dir = path / revol
    if revol_dir.is_dir():
        path = revol_dir

    candidate_files = [f for f in os.listdir(path) if f.startswith(swid)]
    if len(candidate_files) == 0:
        raise FileNotFoundError(f"No PIF file found for SWID {swid} in directory: {path}")
    elif len(candidate_files) > 1:
        raise FileNotFoundError(
            f"Multiple PIF files found for SWID {swid} in directory: {path}\n"
            f"Found: {candidate_files}\n"
            "Please specify the exact file path."
        )
    resolved_path = path / candidate_files[0]
    if not resolved_path.is_file():
        raise FileNotFoundError(f"PIF file not found: {resolved_path}")

    return str(resolved_path)


def resolve_event_path(
    events_path: Optional[Union[str, Path]] = None, swid: Optional[str] = None, config: Optional[Config] = None
) -> str:
    """
    Verify and resolve the events file path.

    Parameters
    ----------
    events_path : str or Path, optional
        File path or directory path containing events file.
        If directory, searches for files containing 'isgri_events'.
    swid : str, optional
        Science Window ID to construct default event path. Requires events_path to be None and defined archive_path in Config.
    config : Config, optional
        Configuration object containing archive_path.

    Returns
    -------
    resolved_path : str
        Resolved absolute path to valid events file.

    Raises
    ------
    FileNotFoundError
        If path doesn't exist, no events file found, or multiple
        candidates found in directory.
    ValueError
        If ISGR-EVTS-ALL extension not found in file.

    Examples
    --------
    >>> # Direct file path
    >>> path = verify_events_path("isgri_events.fits")

    >>> # Directory with single events file
    >>> path = verify_events_path("/data/scw/0234/001200340010.001")

    >>> # Will raise error if multiple files
    >>> verify_events_path("/data/scw/")  # Multiple SCWs present
    FileNotFoundError: Multiple isgri_events files found...

    Notes
    -----
    - If both events_path and swid are provided, events_path takes precedence.
    - If neither is provided, raises ValueError.
    - Using swid requires archive_path in Config and constructs path as:
      {archive_path}/{revol}/{swid}/isgri_events.fits.gz
    """
    if events_path is not None:
        path = Path(events_path)
        if path.is_file():
            return verify_events_file(path)
        elif not path.is_dir():
            raise FileNotFoundError(f"Events path does not exist: {path}")
    elif config is not None and swid is not None:
        if config.archive_path is None:
            raise ValueError("archive_path must be defined in config to resolve events path using swid.")
        path = Path(config.archive_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Archive path from config is not a directory: {path}")
    else:
        raise ValueError("Either events_path must be provided here or in config along with swid.")

    if swid is None:
        candidate_files = [f for f in os.listdir(path) if "isgri_events" in f]
        if len(candidate_files) == 1:
            resolved_path = path / candidate_files[0]
            return verify_events_file(resolved_path)
        elif len(candidate_files) > 1:
            raise FileNotFoundError(
                f"Multiple isgri_events files found in directory: {path}\n"
                f"Found: {candidate_files}\n"
                "Please specify the exact file path."
            )
        else:
            raise FileNotFoundError(f"No isgri_events file found in directory: {path}")

    revol = swid[:4]
    revol_dir = path / revol
    if revol_dir.is_dir():
        path = revol_dir
    candidate_dirs = [d for d in os.listdir(path) if d.startswith(swid) and (path / d).is_dir()]
    if len(candidate_dirs) == 1:
        path = path / candidate_dirs[0]
    elif len(candidate_dirs) > 1:
        raise FileNotFoundError(
            f"Multiple directories found for SWID {swid} in directory: {path}\n"
            f"Found: {candidate_dirs}\n"
            "Please specify the exact file path."
        )
    else:
        raise FileNotFoundError(f"No directory found for SWID {swid} in directory: {path}")

    candidate_files = [f for f in os.listdir(path) if "isgri_events" in f]

    if len(candidate_files) == 0:
        raise FileNotFoundError(f"No isgri_events file found in directory: {path}")
    elif len(candidate_files) > 1:
        raise FileNotFoundError(
            f"Multiple isgri_events files found in directory: {path}\n"
            f"Found: {candidate_files}\n"
            "Please specify the exact file path."
        )
    else:
        resolved_path = str(path / candidate_files[0])

    return verify_events_file(resolved_path)


def verify_events_file(events_path: Union[str, Path]) -> str:
    """
    Verify that the provided events file path is valid.
    Parameters
    ----------
    events_path : str or Path
        Path to events FITS file.
    Returns
    -------
    str
        Validated absolute path to events FITS file.
    Raises
    ------
    FileNotFoundError
        If events file not found.
    ValueError
        If required FITS extension missing.
    """

    try:
        with fits.open(events_path) as hdu:
            if "ISGR-EVTS-ALL" not in hdu:
                raise ValueError(f"Invalid events file: ISGR-EVTS-ALL extension not found in {events_path}")
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Cannot open FITS file {events_path}: {e}")
    return str(events_path)


def load_isgri_events(events_path: Union[str, Path]) -> Tuple[Table, NDArray[np.float64], Dict]:
    """
    Load ISGRI photon events from FITS file.

    Parameters
    ----------
    events_path : str or Path
        Path to events FITS file or directory containing it.

    Returns
    -------
    events : Table
        Astropy Table with columns:
        - TIME : Event time in IJD
        - ISGRI_ENERGY : Energy in keV
        - DETY : Y detector coordinate (0-129)
        - DETZ : Z detector coordinate (0-133)
    gtis : ndarray, shape (N, 2) or None
        Good Time Intervals [start, stop] pairs in IJD.
        If no GTI extension found, returns None.
    metadata : dict
        Header metadata with keys:
        - REVOL : Revolution number
        - SWID : Science Window ID
        - TSTART, TSTOP : Start/stop times (IJD)
        - RA_SCX, DEC_SCX : Pointing axis coordinates
        - RA_SCZ, DEC_SCZ : Z-axis coordinates
        - NoEVTS : Number of events in file

    Raises
    ------
    FileNotFoundError
        If events file not found or invalid.
    ValueError
        If required FITS extension missing.

    Examples
    --------
    >>> events, gtis, meta = load_isgri_events("isgri_events.fits")
    >>> print(f"Loaded {len(events)} events")
    >>> print(f"Time range: {meta['TSTART']:.1f} - {meta['TSTOP']:.1f} IJD")
    >>> print(f"GTIs: {len(gtis)} intervals")

    >>> # Check energy range
    >>> print(f"Energy: {events['ISGRI_ENERGY'].min():.1f} - "
    ...       f"{events['ISGRI_ENERGY'].max():.1f} keV")

    See Also
    --------
    load_isgri_pif : Apply detector response weighting
    """

    with fits.open(events_path) as hdu:
        # Load events
        events_data = hdu["ISGR-EVTS-ALL"].data
        header = hdu["ISGR-EVTS-ALL"].header

        # Extract metadata
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
            "NoEVTS": header.get("NAXIS2"),
        }

        # Load GTIs
        try:
            gti_data = hdu["IBIS-GNRL-GTI"].data
            gtis = np.column_stack([gti_data["START"], gti_data["STOP"]])
        except (KeyError, IndexError):
            # No GTI extension - return empty GTI
            # t_start = events_data["TIME"][0]
            # t_stop = events_data["TIME"][-1]
            # gtis = np.array([[t_start, t_stop]])
            gtis = None

    # Filter bad events (SELECT_FLAG != 0)
    good_mask = events_data["SELECT_FLAG"] == 0
    events = Table(events_data[good_mask])

    return events, gtis, metadata


def default_pif_metadata() -> Dict:
    """
    Create default PIF metadata for cases without PIF file.

    Used when no PIF weighting is applied. All modules are
    considered active with unit weight.

    Returns
    -------
    metadata : dict
        Default PIF metadata with:
        - SWID : None
        - SRC_RA, SRC_DEC : None (no source position)
        - Source_Name : None
        - cod : None (no coding fraction)
        - No_Modules : [True]*8 (all modules active)

    Examples
    --------
    >>> meta = default_pif_metadata()
    >>> print(meta['No_Modules'])
    [True, True, True, True, True, True, True, True]
    """
    return {
        "SWID": None,
        "SRC_RA": None,
        "SRC_DEC": None,
        "Source_Name": None,
        "cod": None,
        "No_Modules": [True] * 8,  # All modules active
    }


def merge_metadata(events_metadata: Dict, pif_metadata: Dict) -> Dict:
    """
    Merge events and PIF metadata dictionaries.

    PIF metadata takes precedence except for SWID, which is
    preserved from events metadata.

    Parameters
    ----------
    events_metadata : dict
        Metadata from events file.
    pif_metadata : dict
        Metadata from PIF file.

    Returns
    -------
    merged : dict
        Combined metadata with PIF values overwriting events values
        (except SWID).

    Examples
    --------
    >>> events_meta = {'SWID': '023400100010', 'TSTART': 3000.0}
    >>> pif_meta = {'SWID': '999999999999', 'SRC_RA': 83.63, 'SRC_DEC': 22.01}
    >>> merged = merge_metadata(events_meta, pif_meta)
    >>> print(merged['SWID'])  # Preserved from events
    023400100010
    >>> print(merged['SRC_RA'])  # From PIF
    83.63
    """
    merged = events_metadata.copy()

    for key, value in pif_metadata.items():
        if key != "SWID":  # Preserve SWID from events
            merged[key] = value

    return merged


def load_isgri_pif(
    pif_path: Union[str, Path],
    events: Table,
    pif_extension: int = -1,
) -> Tuple[Table, NDArray[np.float64], Dict]:
    """
    Load PIF (Pixel Illumination Fraction) file and apply to events.

    Filters events by PIF threshold and returns PIF weights for
    detector response correction.

    Parameters
    ----------
    pif_path : str or Path
        Path to PIF FITS file.
    events : Table
        Events table from load_isgri_events() with DETZ, DETY columns.
    pif_extension : int, default -1
        FITS extension index containing PIF data.
        -1 = last extension (typical).

    Returns
    -------
    filtered_events : Table
        Events with PIF >= threshold.
    pif_weights : ndarray
        PIF value for each filtered event (for response correction).
    pif_metadata : dict
        Metadata with keys:
        - SWID : Science Window ID
        - Source_ID, Source_Name : Source identifiers
        - SRC_RA, SRC_DEC : Source position (degrees)
        - cod : Coding fraction (0.0-1.0)
        - No_Modules : Array of active module flags

    Raises
    ------
    ValueError
        If PIF file has invalid shape (must be 134×130).
    FileNotFoundError
        If PIF file not found.

    Examples
    --------
    >>> events, gtis, meta = load_isgri_events("events.fits")
    >>> filtered, weights, pif_meta = load_isgri_pif(
    ...     "pif_model.fits",
    ...     events,
    ...     pif_threshold=0.5
    ... )
    >>>
    >>> print(f"Filtered: {len(filtered)}/{len(events)} events")
    >>> print(f"Coding fraction: {pif_meta['cod']:.2%}")
    >>> print(f"Active modules: {np.sum(pif_meta['No_Modules'])}/8")

    >>> # Use for light curve with response correction
    >>> from isgri.utils import LightCurve
    >>> lc = LightCurve(
    ...     time=filtered['TIME'],
    ...     energies=filtered['ISGRI_ENERGY'],
    ...     dety=filtered['DETY'],
    ...     detz=filtered['DETZ'],
    ...     weights=weights,
    ...     gtis=gtis,
    ...     metadata={**meta, **pif_meta}
    ... )

    See Also
    --------
    load_isgri_events : Load event data
    coding_fraction : Calculate coded fraction
    """
    pif_path = Path(pif_path)
    if not pif_path.exists():
        raise FileNotFoundError(f"PIF file not found: {pif_path}")

    # Load PIF file
    with fits.open(pif_path) as hdu:
        pif_file = np.array(hdu[pif_extension].data)
        header = hdu[pif_extension].header

    # Validate shape
    if pif_file.shape != (134, 130):
        raise ValueError(
            f"Invalid PIF file shape: expected (134, 130), got {pif_file.shape}. " "File may be empty or corrupted."
        )

    # Extract metadata
    pif_metadata = {
        "SWID": header.get("SWID"),
        "Source_ID": header.get("SOURCEID"),
        "Source_Name": header.get("NAME"),
        "SRC_RA": header.get("RA_OBJ"),
        "SRC_DEC": header.get("DEC_OBJ"),
    }

    # Compute quality metrics
    pif_metadata["cod"] = coding_fraction(pif_file, events)
    pif_metadata["No_Modules"] = estimate_active_modules(pif_file)

    # Apply PIF mask
    pif_weights = pif_file[events["DETZ"], events["DETY"]]

    return events, pif_weights, pif_metadata
