"""
Time Conversion Utilities for INTEGRAL
=======================================

Convert between INTEGRAL Julian Date (IJD) and other time formats.

INTEGRAL uses IJD as its standard time format:
- IJD 0.0 = 2000-01-01 00:00:00 TT (Terrestrial Time)
- IJD = MJD - 51544.0
- IJD uses TT time scale (atomic time)

Functions
---------
ijd2utc : Convert IJD to UTC ISO format
utc2ijd : Convert UTC ISO format to IJD
ijd2mjd : Convert IJD to Modified Julian Date
mjd2ijd : Convert Modified Julian Date to IJD

Examples
--------
>>> from isgri.utils import ijd2utc, utc2ijd
>>>
>>> # Convert IJD to readable format
>>> utc_str = ijd2utc(3000.5)
>>> print(utc_str)
2008-03-17 11:58:55.816
>>>
>>> # Round trip conversion
>>> ijd = utc2ijd("2010-01-01 12:00:00")
>>> print(f"IJD: {ijd:.3f}")
IJD: 3653.500

Notes
-----
- IJD uses TT (Terrestrial Time) scale for consistency with spacecraft clock
- UTC conversions account for leap seconds automatically
- Precision: ~1 microsecond for typical astronomical observations
"""

from astropy.time import Time
from typing import Union
import numpy as np
from numpy.typing import NDArray

# INTEGRAL epoch: MJD 51544.0 = 2000-01-01 00:00:00 TT
INTEGRAL_EPOCH_MJD = 51544.0


def ijd2utc(ijd_time: Union[float, NDArray[np.float64]]) -> Union[str, NDArray[np.str_]]:
    """
    Convert IJD (INTEGRAL Julian Date) to UTC ISO format.

    Parameters
    ----------
    ijd_time : float or ndarray
        IJD time value(s). Can be scalar or array.

    Returns
    -------
    utc_str : str or ndarray of str
        UTC time in ISO format 'YYYY-MM-DD HH:MM:SS.sss'
        Scalar if input is scalar, array if input is array.

    Examples
    --------
    >>> ijd2utc(0.0)
    '1999-12-31 23:58:55.816'

    >>> ijd2utc(1000.5)
    '2002-09-27 11:58:55.816'

    >>> # Array conversion
    >>> ijds = np.array([0.0, 1000.0, 2000.0])
    >>> utcs = ijd2utc(ijds)
    >>> print(utcs[0])
    1999-12-31 23:58:55.816

    See Also
    --------
    utc2ijd : Inverse conversion (UTC to IJD)
    ijd2mjd : Convert IJD to MJD

    Notes
    -----
    - IJD 0.0 corresponds to 2000-01-01 00:00:00 TT
    - Accounts for leap seconds via UTC scale
    - Output precision is milliseconds (3 decimal places)
    """
    mjd = ijd_time + INTEGRAL_EPOCH_MJD
    t = Time(mjd, format="mjd", scale="tt")
    return t.utc.iso


def utc2ijd(utc_time: Union[str, NDArray[np.str_]]) -> Union[float, NDArray[np.float64]]:
    """
    Convert UTC ISO format to IJD (INTEGRAL Julian Date).

    Parameters
    ----------
    utc_time : str or ndarray of str
        UTC time in ISO format. Accepts:
        - 'YYYY-MM-DD HH:MM:SS' (space separator)
        - 'YYYY-MM-DDTHH:MM:SS' (T separator)
        - Can include fractional seconds

    Returns
    -------
    ijd_time : float or ndarray
        IJD time value(s)

    Examples
    --------
    >>> utc2ijd('1999-12-31 23:58:55.816')
    0.0

    >>> utc2ijd('2002-09-27 00:00:00')
    1000.0

    >>> # ISO 8601 format (with T separator)
    >>> utc2ijd('2010-01-01T12:00:00')
    3653.5

    >>> # Array conversion
    >>> utcs = ['2000-01-01 00:00:00', '2000-01-02 00:00:00']
    >>> ijds = utc2ijd(utcs)
    >>> print(ijds)
    [0.00074287 1.00074287]

    See Also
    --------
    ijd2utc : Inverse conversion (IJD to UTC)
    mjd2ijd : Convert MJD to IJD

    Notes
    -----
    - Automatically handles both space and T separators
    - Accounts for leap seconds
    - Returns TT (Terrestrial Time) scale
    """
    # Handle T separator in ISO 8601 format
    if isinstance(utc_time, str):
        utc_time = utc_time.replace("T", " ")
    elif isinstance(utc_time, np.ndarray):
        # Vectorized string replacement for arrays
        utc_time = np.char.replace(utc_time, "T", " ")

    t = Time(utc_time, format="iso", scale="utc")
    return t.tt.mjd - INTEGRAL_EPOCH_MJD


def ijd2mjd(ijd_time: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
    """
    Convert IJD to Modified Julian Date (MJD).

    Simple offset: MJD = IJD + 51544.0

    Parameters
    ----------
    ijd_time : float or ndarray
        IJD time value(s)

    Returns
    -------
    mjd_time : float or ndarray
        MJD time value(s)

    Examples
    --------
    >>> ijd2mjd(0.0)
    51544.0

    >>> ijd2mjd(3653.5)
    55197.5

    See Also
    --------
    mjd2ijd : Inverse conversion
    """
    return ijd_time + INTEGRAL_EPOCH_MJD


def mjd2ijd(mjd_time: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
    """
    Convert Modified Julian Date (MJD) to IJD.

    Simple offset: IJD = MJD - 51544.0

    Parameters
    ----------
    mjd_time : float or ndarray
        MJD time value(s)

    Returns
    -------
    ijd_time : float or ndarray
        IJD time value(s)

    Examples
    --------
    >>> mjd2ijd(51544.0)
    0.0

    >>> mjd2ijd(55197.5)
    3653.5

    See Also
    --------
    ijd2mjd : Inverse conversion
    """
    return mjd_time - INTEGRAL_EPOCH_MJD
