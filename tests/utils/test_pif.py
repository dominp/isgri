import pytest
import numpy as np
from isgri.utils.pif import select_isgri_module, estimate_active_modules


def test_select_isgri_module():
    """Test module selection returns correct boundaries."""
    # Module 0 (first row, first column)
    x1, x2, y1, y2 = select_isgri_module(0)
    assert x1 == 0 and x2 == 32
    assert y1 == 0 and y2 == 64

    # Module 7 (last row, second column)
    x1, x2, y1, y2 = select_isgri_module(7)
    assert x1 == 100 and x2 == 134
    assert y1 == 64 and y2 == 130


def test_estimate_active_modules():
    """Test active module estimation."""
    # All active
    mask_full = np.ones((134, 130))
    mods = estimate_active_modules(mask_full)
    assert len(mods) == 8
    assert np.sum(mods) == 8

    # All inactive
    mask_empty = np.zeros((134, 130))
    mods = estimate_active_modules(mask_empty)
    assert np.sum(mods) == 0
