"""
Basic example testing the adaptER-covid19 personal bankruptcy models
"""

import sys

from adapter_covid19.datasources import Reader
from adapter_covid19.enums import Region
from adapter_covid19.personal_insolvency import PersonalBankruptcyModel
from tests.adapter_covid19.utilities import (
    DATA_PATH,
    ALL_UTILISATIONS,
    UTILISATION_NO_COVID_NO_LOCKDOWN,
    state_from_utilisation,
    advance_state,
)

sys.path.append("src/adapter_covid19")


def pytest_generate_tests(metafunc):
    if "personal_bankruptcy_model_cls" in metafunc.fixturenames:
        metafunc.parametrize("personal_bankruptcy_model_cls", [PersonalBankruptcyModel])
    if "utilisation" in metafunc.fixturenames:
        metafunc.parametrize("utilisation", ALL_UTILISATIONS)


class TestClass:
    def test_interface(self, personal_bankruptcy_model_cls, utilisation):
        reader = Reader(DATA_PATH)
        state = state_from_utilisation(UTILISATION_NO_COVID_NO_LOCKDOWN)
        new_state = advance_state(state, utilisation)
        pb_model = personal_bankruptcy_model_cls()
        pb_model.load(reader)
        pb_model.simulate(state)
        pb_model.simulate(new_state)
        for region in Region:
            assert 0 <= new_state.personal_state.personal_bankruptcy[region] <= 1
