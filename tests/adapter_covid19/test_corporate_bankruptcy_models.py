"""
Basic example testing the adaptER-covid19 corporate bankruptcy models
"""
import itertools
import sys

from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel
from adapter_covid19.data_structures import IoGdpState
from adapter_covid19.datasources import Reader
from adapter_covid19.enums import PrimaryInput, Region, Sector, Age
from tests.adapter_covid19.utilities import (
    DATA_PATH,
    ALL_UTILISATIONS,
    state_from_utilisation,
    UTILISATION_NO_COVID_NO_LOCKDOWN,
    advance_state,
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
    if "utilisation" in metafunc.fixturenames:
        metafunc.parametrize(
            # TODO: figure out how to test this properly
            "utilisation",
            ALL_UTILISATIONS,
        )


class TestClass:
    def test_interface(self, corporate_bankruptcy_model_cls, utilisation):
        reader = Reader(DATA_PATH)
        state = state_from_utilisation(UTILISATION_NO_COVID_NO_LOCKDOWN)
        state.gdp_state = IoGdpState(primary_inputs=DUMMY_PRIMARY_INPUTS)
        new_state = advance_state(state, utilisation)
        new_state.gdp_state = IoGdpState(primary_inputs=DUMMY_PRIMARY_INPUTS)
        cb_model = corporate_bankruptcy_model_cls()
        cb_model.load(reader)
        cb_model.simulate(state)
        cb_model.simulate(new_state)
        for (
            _business_size,
            mapping,
        ) in new_state.corporate_state.proportion_solvent.items():
            for sector, solvent in mapping.items():
                assert 0 <= solvent <= 1
