"""
Basic example testing the adaptER-covid19 GDP models
"""

import sys

from adapter_covid19.datasources import Reader
from adapter_covid19.gdp import LinearGdpModel, SupplyDemandGdpModel
from tests.adapter_covid19.utilities import (
    MAX_UTILISATIONS,
    MIN_UTILISATIONS,
    DATA_PATH,
)

sys.path.append("src/adapter_covid19")


def pytest_generate_tests(metafunc):
    if "gdp_model_cls" in metafunc.fixturenames:
        metafunc.parametrize("gdp_model_cls", [LinearGdpModel, SupplyDemandGdpModel])
    if "utilisations" in metafunc.fixturenames:
        metafunc.parametrize("utilisations", [MAX_UTILISATIONS, MIN_UTILISATIONS])
    if "lockdown" in metafunc.fixturenames:
        metafunc.parametrize("lockdown", [True, False])


class TestClass:
    def test_interface(self, gdp_model_cls, lockdown, utilisations):
        reader = Reader(DATA_PATH)
        gdp_model = gdp_model_cls()
        gdp_model.load(reader)
        gdp_model.simulate(
            time=0, lockdown=lockdown, lockdown_exit_time=0, utilisations=utilisations
        )
        assert 0 <= sum(gdp_model.results.gdp[0].values()) <= gdp_model.results.max_gdp
