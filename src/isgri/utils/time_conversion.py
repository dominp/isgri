from astropy.time import Time


def ijd2utc(t):
    """
    Converts IJD (INTEGRAL Julian Date) time to UTC ISO format.

    Args:
        t (float or ndarray): IJD time value(s).

    Returns:
        str or ndarray: UTC time in ISO format (YYYY-MM-DD HH:MM:SS.sss).

    Examples:
        >>> ijd2utc(0.0)
        '1999-12-31 23:58:55.817'
        >>> ijd2utc(1000.5)
        '2002-09-27 11:58:55.816'
    """
    return Time(t + 51544, format="mjd", scale="tt").utc.iso


def utc2ijd(t):
    """
    Converts UTC ISO format time to IJD (INTEGRAL Julian Date).

    Args:
        t (str or ndarray): UTC time in ISO format (YYYY-MM-DD HH:MM:SS).

    Returns:
        float or ndarray: IJD time value(s).

    Examples:
        >>> utc2ijd('1999-12-31 23:58:55.817')
        0.0
        >>> utc2ijd('2002-09-27 00:00:00')
        1000.0
    """
    return Time(t, format="iso", scale="utc").tt.mjd - 51544
