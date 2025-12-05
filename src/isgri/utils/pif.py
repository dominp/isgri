"""
ISGRI Detector Pixel Illumination Fraction (PIF) Tools
========================================================

Functions for working with ISGRI detector response maps (PIF files).
PIF values indicate what fraction of source flux reaches each detector pixel,
accounting for shadowing by the coded mask.

PIF values range from 0 (fully shadowed) to 1 (fully illuminated).

Functions
---------
select_isgri_module : Get detector coordinate bounds for a module
apply_pif_mask : Filter events by PIF threshold
coding_fraction : Calculate coded fraction for source position
estimate_active_modules : Determine which modules have significant PIF coverage

Examples
--------
>>> import numpy as np
>>> from astropy.table import Table
>>>
>>> # Load PIF and events
>>> pif_file = np.random.rand(134, 130)  # Mock PIF
>>> events = Table({'DETZ': [10, 20], 'DETY': [15, 25]})
>>>
>>> # Apply PIF weighting
>>> filtered_events, weights = apply_pif_mask(pif_file, events, pif_threshold=0.5)
>>>
>>> # Check which modules are active
>>> active = estimate_active_modules(pif_file)
>>> print(f"Active modules: {np.where(active)[0]}")
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from astropy.table import Table

# ISGRI detector module boundaries (mm coordinates)
# 8 modules total: 4 rows × 2 columns
# Z-axis (rows): 4 modules 
# Y-axis (cols): 2 modules
DETZ_BOUNDS = [0, 32, 66, 100, 134]  # 5 boundaries for 4 rows
DETY_BOUNDS = [0, 64, 130]  # 3 boundaries for 2 columns


def select_isgri_module(module_no: int) -> Tuple[int, int, int, int]:
    """
    Get detector coordinate bounds for specified module.

    ISGRI has 8 modules arranged in 4 rows × 2 columns:

    Module layout:
        [0] [1]
        [2] [3]
        [4] [5]
        [6] [7]

    Parameters
    ----------
    module_no : int
        Module number (0-7)

    Returns
    -------
    z1, z2, y1, y2 : int
        Detector coordinate bounds (DETZ min/max, DETY min/max)

    Raises
    ------
    ValueError
        If module_no not in range [0, 7]

    Examples
    --------
    >>> z1, z2, y1, y2 = select_isgri_module(0)
    >>> print(f"Module 0: DETZ=[{z1},{z2}], DETY=[{y1},{y2}]")
    Module 0: DETZ=[0,32], DETY=[0,64]

    >>> # Module 3 is bottom-right
    >>> select_isgri_module(3)
    (66, 100, 64, 130)
    """
    if not (0 <= module_no <= 7):
        raise ValueError(f"module_no must be in [0, 7], got {module_no}")

    col = module_no % 2  # 0=left, 1=right
    row = module_no // 2  # 0-3 from top to bottom

    z1, z2 = DETZ_BOUNDS[row], DETZ_BOUNDS[row + 1]
    y1, y2 = DETY_BOUNDS[col], DETY_BOUNDS[col + 1]

    return z1, z2, y1, y2


def apply_pif_mask(
    pif_file: NDArray[np.float64],
    events: Table,
    pif_threshold: float = 0.5,
) -> Tuple[Table, NDArray[np.float64]]:
    """
    Filter events by PIF threshold and return PIF weights.

    Events with PIF < threshold are removed. Remaining events are
    weighted by their PIF values for response correction.

    Parameters
    ----------
    pif_file : ndarray, shape (134, 130)
        2D PIF array (DETZ x DETY coordinates)
    events : Table
        Event table with 'DETZ' and 'DETY' columns
    pif_threshold : float, default 0.5
        Minimum PIF value to keep event (0.0-1.0)

    Returns
    -------
    filtered_events : Table
        Events with PIF >= threshold
    pif_weights : ndarray
        PIF value for each filtered event

    Raises
    ------
    ValueError
        If pif_threshold not in [0, 1]
        If events missing 'DETZ' or 'DETY' columns
        If PIF dimensions don't match expected (134, 130)

    Examples
    --------
    >>> pif = np.random.rand(134, 130)
    >>> events = Table({'DETZ': [10, 20, 30], 'DETY': [15, 25, 35]})
    >>>
    >>> # Keep only well-illuminated events
    >>> filtered, weights = apply_pif_mask(pif, events, pif_threshold=0.7)
    >>> print(f"Kept {len(filtered)}/{len(events)} events")
    >>> print(f"Mean weight: {weights.mean():.3f}")
    """
    # Validate inputs
    if not (0 <= pif_threshold <= 1):
        raise ValueError(f"pif_threshold must be in [0, 1], got {pif_threshold}")

    if pif_file.shape != (134, 130):
        raise ValueError(f"PIF file must have shape (134, 130), got {pif_file.shape}")

    if "DETZ" not in events.colnames or "DETY" not in events.colnames:
        raise ValueError("Events table must have 'DETZ' and 'DETY' columns")

    # Create mask for events above threshold
    pif_filter = pif_file > pif_threshold

    # Get PIF values at event positions
    event_pif = pif_file[events["DETZ"], events["DETY"]]

    # Apply filter
    mask = event_pif > pif_threshold
    filtered_events = events[mask]
    pif_weights = event_pif[mask]

    return filtered_events, pif_weights


def coding_fraction(
    pif_file: NDArray[np.float64],
    events: Table,
) -> float:
    """
    Calculate fraction of detector that is fully coded.

    Uses events with PIF=1.0 (fully illuminated) to estimate
    the size of the fully coded field of view.

    Parameters
    ----------
    pif_file : ndarray, shape (134, 130)
        2D PIF array
    events : Table
        Event table with 'DETZ' and 'DETY' columns

    Returns
    -------
    coding_fraction : float
        Fraction of detector area that is fully coded (0.0-1.0)

    Notes
    -----
    Fully coded region has PIF=1.0 for on-axis sources.
    Partially coded region has 0 < PIF < 1.

    Examples
    --------
    >>> pif = np.ones((134, 130))
    >>> pif[50:80, 40:90] = 1.0  # Fully coded region
    >>> events = Table({'DETZ': np.arange(134), 'DETY': np.arange(130)})
    >>>
    >>> frac = coding_fraction(pif, events)
    >>> print(f"Coding fraction: {frac:.2%}")
    """
    if pif_file.shape != (134, 130):
        raise ValueError(f"PIF must have shape (134, 130), got {pif_file.shape}")

    if "DETZ" not in events.colnames or "DETY" not in events.colnames:
        raise ValueError("Events must have 'DETZ' and 'DETY' columns")

    # Find fully coded pixels (PIF = 1.0)
    fully_coded = pif_file == 1.0

    # Get events in fully coded region
    coded_events = events[fully_coded[events["DETZ"], events["DETY"]]]

    if len(coded_events) == 0:
        return 0.0

    # Calculate extent in Y and Z
    dety_range = np.max(coded_events["DETY"]) - np.min(coded_events["DETY"])
    detz_range = np.max(coded_events["DETZ"]) - np.min(coded_events["DETZ"])

    # Normalize by detector size (Y: 0-129, Z: 0-133)
    frac_y = dety_range / 129.0
    frac_z = detz_range / 133.0

    # Area fraction
    coding_frac = frac_y * frac_z

    return coding_frac


def estimate_active_modules(
    pif_file: NDArray[np.float64],
    threshold: float = 0.2,
) -> NDArray[np.bool_]:
    """
    Determine which detector modules have significant PIF coverage.

    A module is considered active if more than `threshold` fraction
    of its pixels have PIF > 0.01.

    Parameters
    ----------
    pif_file : ndarray, shape (134, 130)
        2D PIF array
    threshold : float, default 0.2
        Minimum fraction of illuminated pixels (0.0-1.0)

    Returns
    -------
    active_modules : ndarray of bool, shape (8,)
        True if module is active, False otherwise

    Examples
    --------
    >>> pif = np.random.rand(134, 130)
    >>> pif[:50, :] = 0  # Top modules dark
    >>>
    >>> active = estimate_active_modules(pif, threshold=0.2)
    >>> print(f"Active modules: {np.where(active)[0]}")
    Active modules: [2 3 4 5 6 7]

    >>> # Get list of active module numbers
    >>> active_list = np.where(active)[0].tolist()
    """
    if pif_file.shape != (134, 130):
        raise ValueError(f"PIF must have shape (134, 130), got {pif_file.shape}")

    if not (0 <= threshold <= 1):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    active_modules = np.zeros(8, dtype=bool)

    for module_no in range(8):
        z1, z2, y1, y2 = select_isgri_module(module_no)

        # Get PIF values for this module
        module_pif = pif_file[z1:z2, y1:y2].flatten()

        # Count illuminated pixels (PIF > 0.01)
        n_illuminated = np.sum(module_pif > 0.01)
        n_total = len(module_pif)

        # Check if fraction exceeds threshold
        if n_illuminated / n_total > threshold:
            active_modules[module_no] = True

    return active_modules
