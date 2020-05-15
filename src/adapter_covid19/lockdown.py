import functools
import itertools

from adapter_covid19.enums import Region, Sector, Age

from adapter_covid19.datasources import (
    Reader,
    SectorDataSource,
    RegionSectorAgeDataSource,
)


@functools.lru_cache()
def _base_lockdown_state(data_path: str) -> float:
    """
    Get proportion of workforce at work during lockdown

    :param data_path: path to economics data
    :return:
    """
    reader = Reader(data_path)
    keyworker = SectorDataSource("keyworker").load(reader)
    workers = RegionSectorAgeDataSource("workers").load(reader)
    return sum(
        keyworker[s] * workers[r, s, a]
        for r, s, a in itertools.product(Region, Sector, Age)
    ) / sum(workers.values())


def get_lockdown_factor(
    lockdown: bool, slow_unlock: bool, lockdown_exit_time: int, time: int,
) -> float:
    """
    Get how locked-down the country is
    0 -> no lockdown
    1 -> fully locked down

    :param lockdown:
    :param lockdown_exit_time:
    :param time:
    :return:
    """
    if lockdown:
        return 1
    if not slow_unlock or not lockdown_exit_time:
        # i.e. we've never been in a lockdown
        return 0
    assert time >= lockdown_exit_time, (time, lockdown_exit_time)
    # Send 10% of the remaining people back every 10 days
    n = 1 + (time - lockdown_exit_time) // 10
    return 0.9 ** n


def get_working_factor(data_path: str, lockdown_factor: float) -> float:
    """
    Get proportion of people working

    :param data_path:
    :param lockdown_factor:
    :return:
    """
    base_lockdown = _base_lockdown_state(data_path)
    return base_lockdown + (1 - base_lockdown) * (1 - lockdown_factor)
