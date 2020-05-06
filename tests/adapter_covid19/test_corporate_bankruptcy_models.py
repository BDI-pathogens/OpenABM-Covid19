"""
Basic example testing the adaptER-covid19 corporate bankruptcy models
"""
import itertools
import sys

from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel
from adapter_covid19.data_structures import SimulateState, IoGdpState
from adapter_covid19.datasources import Reader
from adapter_covid19.enums import PrimaryInput, Region, Sector, Age
from tests.adapter_covid19.utilities import (
    DATA_PATH,
    MIN_UTILISATIONS,
    MAX_UTILISATIONS,
    FLAT_UTILISATIONS,
)

sys.path.append("src/adapter_covid19")

DUMMY_PRIMARY_INPUTS = {
    (PrimaryInput.NET_OPERATING_SURPLUS, r, s, a): 100
    for r, s, a in itertools.product(Region, Sector, Age)
}


def pytest_generate_tests(metafunc):
    if "corporate_bankruptcy_model_cls" in metafunc.fixturenames:
        metafunc.parametrize(
            "corporate_bankruptcy_model_cls", [CorporateBankruptcyModel]
        )
    if "utilisations" in metafunc.fixturenames:
        metafunc.parametrize(
            # TODO: figure out how to test this properly
            "utilisations",
            [MIN_UTILISATIONS, MAX_UTILISATIONS, FLAT_UTILISATIONS],
        )


class TestClass:
    def test_interface(self, corporate_bankruptcy_model_cls, utilisations):
        reader = Reader(DATA_PATH)
        state_0 = SimulateState(
            time=0,
            lockdown=False,
            utilisations=utilisations,
            gdp_state=IoGdpState(primary_inputs=DUMMY_PRIMARY_INPUTS),
        )
        state_1 = SimulateState(
            time=1,
            lockdown=True,
            utilisations=utilisations,
            gdp_state=IoGdpState(primary_inputs=DUMMY_PRIMARY_INPUTS),
        )
        cb_model = corporate_bankruptcy_model_cls()
        cb_model.load(reader)
        cb_model.simulate(state_0)
        cb_model.simulate(state_1)
        for discount in state_1.corporate_state.gdp_discount_factor.values():
            assert 0 <= discount <= 1
