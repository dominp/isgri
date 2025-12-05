from astropy.time import Time


def ijd2utc(ijd_time):
    """
    Converts IJD (INTEGRAL Julian Date) time to UTC ISO format.

    Args:
        ijd_time (float or ndarray): IJD time value(s).

    Returns:
        str or ndarray: UTC time in ISO format (YYYY-MM-DD HH:MM:SS.sss).

    Examples:
        >>> ijd2utc(0.0)
        '1999-12-31 23:58:55.817'
        >>> ijd2utc(1000.5)
        '2002-09-27 11:58:55.816'
    """
    return Time(ijd_time + 51544, format="mjd", scale="tt").utc.iso


def utc2ijd(utc_time):
    """
    Converts UTC ISO format time to IJD (INTEGRAL Julian Date).

    Args:
        utc_time (str or ndarray): UTC time in ISO format (YYYY-MM-DD HH:MM:SS).

    Returns:
        float or ndarray: IJD time value(s).

    Examples:
        >>> utc2ijd('1999-12-31 23:58:55.817')
        0.0
        >>> utc2ijd('2002-09-27 00:00:00')
        1000.0
    """
    if isinstance(utc_time, str):
        utc_time = utc_time.replace("T", " ")
    return Time(utc_time, format="iso", scale="utc").tt.mjd - 51544
