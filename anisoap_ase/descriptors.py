import numpy as np
def anisoap_stub_descriptor(atoms):
    return np.asarray(atoms.numbers, dtype=np.float64)
