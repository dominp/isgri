from astropy.io import fits
from .pif import apply_pif_mask, coding_fraction, estimate_active_modules
import numpy as np
import os


def verify_events_path(path):
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


def load_events_file(events_path):
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
    return {
        "SWID": None,
        "SRC_RA": None,
        "SRC_DEC": None,
        "Source_Name": None,
        "cod": None,
        "No_Modules": 8,
    }


def merge_metadata(events_metadata, pif_metadata):
    merged_metadata = events_metadata.copy()
    for key in pif_metadata:
        if key == "SWID":
            continue
        merged_metadata[key] = pif_metadata[key]
    return merged_metadata


def load_pif_file(pif_path, events, pif_threshold=0.5, pif_extension=-1):
    with fits.open(pif_path) as hdu:
        pif_file = np.array(hdu[pif_extension].data)
        header = hdu[pif_extension].header

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
