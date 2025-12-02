import pytest
import numpy as np
from isgri.utils import *


def test_times():
    ijd_times = np.random.randint(0, 9001, size=1000) + np.random.random(size=1000)
    utc_times = ijd2utc(ijd_times)
    assert np.allclose(utc2ijd(utc_times), ijd_times)

    start_date = "1999-12-31 23:58:55.817"
    ijd_start_date = utc2ijd(start_date)
    assert np.isclose(ijd_start_date, 0.0, atol=1e-6)

    negative_date = "1985-01-01 00:00:00.000"
    ijd_negative_date = utc2ijd(negative_date)
    assert ijd_negative_date < 0


    
