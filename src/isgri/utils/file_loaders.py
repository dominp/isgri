from astropy.io import fits
from .pif import apply_pif_mask, coding_fraction, estimate_active_modules
import numpy as np
import os


def verify_events_path(path):
    """
    Verifies and resolves the events file path.

    Args:
        path (str): File path or directory path containing events file.

    Returns:
        str: Resolved path to valid events file.

    Raises:
        FileNotFoundError: If path doesn't exist, no events file found, or multiple events files found.
        ValueError: If ISGR-EVTS-ALL extension not found in file.

    """
    if os.path.isfile(path):
        resolved_path = path
    elif os.path.isdir(path):
        candidate_files = [f for f in os.listdir(path) if "isgri_events" in f]
        if len(candidate_files) == 0:
            raise FileNotFoundError("No isgri_events file found in the provided directory.")
        elif len(candidate_files) > 1:
            raise FileNotFoundError(
                f"Multiple isgri_events files found in the provided directory: {path}.",
                "\nPlease specify the exact file paths.",
            )
        else:
            resolved_path = os.path.join(path, candidate_files[0])
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

    with fits.open(resolved_path) as hdu:
        if "ISGR-EVTS-ALL" not in hdu:
            raise ValueError(f"Invalid events file: ISGR-EVTS-ALL extension not found in {resolved_path}")
    return resolved_path


def load_isgri_events(events_path):
    """
    Loads ISGRI events from FITS file.

    Args:
        events_path (str): Path to events file or directory.

    Returns:
        tuple: (events, gtis, metadata) where:
            - events: Structured numpy array with TIME, ISGRI_ENERGY, DETY, DETZ fields
            - gtis: (N, 2) array of Good Time Interval [start, stop] pairs (IJD)
            - metadata: Dictionary with header info (REVOL, SWID, TSTART, etc.)

    """
    confirmed_path = verify_events_path(events_path)
    with fits.open(confirmed_path) as hdu:
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


def default_pif_metadata():
    """
    Creates default PIF metadata dictionary for cases without PIF file.

    Returns:
        dict: Default PIF metadata with all 8 modules active, no source info.
    """
    return {
        "SWID": None,
        "SRC_RA": None,
        "SRC_DEC": None,
        "Source_Name": None,
        "cod": None,
        "No_Modules": 8,
    }


def merge_metadata(events_metadata, pif_metadata):
    """
    Merges events and PIF metadata dictionaries.

    Args:
        events_metadata (dict): Metadata from events file.
        pif_metadata (dict): Metadata from PIF file.

    Returns:
        dict: Combined metadata (PIF metadata overwrites events metadata except SWID).

    """
    merged_metadata = events_metadata.copy()
    for key in pif_metadata:
        if key == "SWID":
            continue
        merged_metadata[key] = pif_metadata[key]
    return merged_metadata


def load_isgri_pif(pif_path, events, pif_threshold=0.5, pif_extension=-1):
    """
    Loads ISGRI PIF (Pixel Illumination Fraction) file and applies mask to events.

    Args:
        pif_path (str): Path to PIF FITS file.
        events (ndarray): Events array from load_isgri_events().
        pif_threshold (float, optional): PIF threshold value (0-1). Defaults to 0.5.
        pif_extension (int, optional): PIF file extension index. Defaults to -1.

    Returns:
        tuple: (piffed_events, pif, metadata_pif) where:
            - piffed_events: Filtered events array with PIF mask applied
            - pif: PIF values for filtered events
            - metadata_pif: Dictionary with source info, coding fraction, active modules
    Raises:
        ValueError: If PIF file shape is invalid. Usually indicates empty or corrupted file.

    """
    with fits.open(pif_path) as hdu:
        pif_file = np.array(hdu[pif_extension].data)
        header = hdu[pif_extension].header

    if pif_file.shape != (134, 130):
        raise ValueError(f"Invalid PIF file shape: expected (134, 130), got {pif_file.shape}")

    metadata_pif = {
        "SWID": header.get("SWID"),
        "Source_ID": header.get("SOURCEID"),
        "Source_Name": header.get("NAME"),
        "SRC_RA": header.get("RA_OBJ"),
        "SRC_DEC": header.get("DEC_OBJ"),
    }
    metadata_pif["cod"] = coding_fraction(pif_file, events)
    metadata_pif["No_Modules"] = estimate_active_modules(pif_file)

    piffed_events, pif = apply_pif_mask(pif_file, events, pif_threshold)

    return piffed_events, pif, metadata_pif
