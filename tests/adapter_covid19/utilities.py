import copy
import itertools

from adapter_covid19.enums import Region, Sector, Age, LabourState

BASE_UTILISATIONS = {k: 0 for k in itertools.product(LabourState, Region, Sector, Age)}
MAX_UTILISATIONS = copy.deepcopy(BASE_UTILISATIONS)
MIN_UTILISATIONS = copy.deepcopy(BASE_UTILISATIONS)
for r, s, a in itertools.product(Region, Sector, Age):
    MAX_UTILISATIONS[LabourState.WORKING, r, s, a] = 1
    MIN_UTILISATIONS[LabourState.ILL, r, s, a] = 1
MAX_SECTOR_UTILISATIONS = {k: 1 for k in Sector}
MIN_SECTOR_UTILISATIONS = {k: 0 for k in Sector}

DATA_PATH = "tests/adapter_covid19/data"
