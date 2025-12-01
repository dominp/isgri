from astropy.time import Time


# time functions
def ijd2utc(t):
    return Time(t + 51544, format="mjd", scale="tt").utc.iso


def utc2ijd(t):
    return Time(t, format="iso", scale="utc").tt.mjd - 51544