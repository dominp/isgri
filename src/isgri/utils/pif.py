import numpy as np

DETZ_BOUNDS, DETY_BOUNDS = [0, 32, 66, 100, 134], [0, 64, 130]  # Detector module boundaries


def select_isgri_module(module_no):
    col = 0 if module_no % 2 == 0 else 1
    row = module_no // 2
    x1, x2 = DETZ_BOUNDS[row], DETZ_BOUNDS[row + 1]
    y1, y2 = DETY_BOUNDS[col], DETY_BOUNDS[col + 1]
    return x1, x2, y1, y2


def apply_pif_mask(pif_file, events, pif_threshold=0.5):
    pif_filter = pif_file > pif_threshold
    piffed_events = events[pif_filter[events["DETZ"], events["DETY"]]]
    pif = pif_file[piffed_events["DETZ"], piffed_events["DETY"]]
    return piffed_events, pif


def coding_fraction(pif_file, events):
    pif_cod = pif_file == 1
    pif_cod = events[pif_cod[events["DETZ"], events["DETY"]]]
    cody = (np.max(pif_cod["DETY"]) - np.min(pif_cod["DETY"])) / 129
    codz = (np.max(pif_cod["DETZ"]) - np.min(pif_cod["DETZ"])) / 133
    pif_cod = codz * cody
    return pif_cod


def estimate_active_modules(mask):
    m, n = DETZ_BOUNDS, DETY_BOUNDS  # Separate modules
    mods = []
    for module_no in range(8):
        x1, x2, y1, y2 = select_isgri_module(module_no)
        a = mask[x1:x2, y1:y2].flatten()
        if len(a[a > 0.01]) / len(a) > 0.2:
            mods.append(1)
        else:
            mods.append(0)
    mods = np.array(mods)
    return mods
