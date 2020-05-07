import itertools

from adapter_covid19.enums import Region, Sector, Age

MAX_UTILISATIONS = {k: 1 for k in itertools.product(Region, Sector, Age)}
MIN_UTILISATIONS = {k: 0 for k in itertools.product(Region, Sector, Age)}
MAX_SECTOR_UTILISATIONS = {k: 1 for k in Sector}
MIN_SECTOR_UTILISATIONS = {k: 0 for k in Sector}

DATA_PATH = "tests/adapter_covid19/data"
