import numpy as np


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
    m, n = [0, 32, 66, 100, 134], [0, 64, 130]  # Separate modules
    mods = []
    for x1, x2 in zip(m[:-1], m[1:]):
        for y1, y2 in zip(n[:-1], n[1:]):
            a = mask[x1:x2, y1:y2].flatten()
            if len(a[a > 0.01]) / len(a) > 0.2:
                mods.append(1)
            else:
                mods.append(0)
    mods = np.array(mods)
    return mods
