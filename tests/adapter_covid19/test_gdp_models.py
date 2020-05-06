"""
Basic example testing the adaptER-covid19 GDP models
"""

import sys

from adapter_covid19.data_structures import SimulateState
from adapter_covid19.datasources import Reader
from adapter_covid19.gdp import (
    LinearGdpModel,
    SupplyDemandGdpModel,
    PiecewiseLinearCobbDouglasGdpModel,
)
from tests.adapter_covid19.utilities import (
    MAX_UTILISATIONS,
    MIN_UTILISATIONS,
    DATA_PATH,
    FLAT_UTILISATIONS,
)

sys.path.append("src/adapter_covid19")


def pytest_generate_tests(metafunc):
    if "gdp_model_cls" in metafunc.fixturenames:
        metafunc.parametrize(
            "gdp_model_cls",
            [LinearGdpModel, SupplyDemandGdpModel, PiecewiseLinearCobbDouglasGdpModel],
        )
    if "utilisations" in metafunc.fixturenames:
        metafunc.parametrize(
            "utilisations", [MAX_UTILISATIONS, MIN_UTILISATIONS, FLAT_UTILISATIONS]
        )


class TestClass:
    def test_interface(self, gdp_model_cls, utilisations):
        reader = Reader(DATA_PATH)
        state = SimulateState(time=0, lockdown=False, utilisations=utilisations)
        gdp_model = gdp_model_cls()
        gdp_model.load(reader)
        gdp_model.simulate(state)
        # Factor of 1.1 is because of the GDP backbone model
        assert 0 <= sum(state.gdp_state.gdp.values()) <= state.gdp_state.max_gdp * 1.1
