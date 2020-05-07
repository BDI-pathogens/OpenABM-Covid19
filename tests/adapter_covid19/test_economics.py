"""
Basic example testing the adaptER-covid19 economics class
"""

import sys

from adapter_covid19.corporate_bankruptcy import (
    CorporateBankruptcyModel,
    NaiveCorporateBankruptcyModel,
)
from adapter_covid19.datasources import Reader
from adapter_covid19.economics import Economics
from adapter_covid19.enums import Region
from adapter_covid19.gdp import PiecewiseLinearCobbDouglasGdpModel
from adapter_covid19.personal_insolvency import PersonalBankruptcyModel
from tests.adapter_covid19.utilities import (
    DATA_PATH,
    state_from_utilisation,
    UTILISATION_NO_COVID_NO_LOCKDOWN,
    advance_state,
    ALL_UTILISATIONS,
)

sys.path.append("src/adapter_covid19")


def pytest_generate_tests(metafunc):
    if "gdp_model_cls" in metafunc.fixturenames:
        metafunc.parametrize("gdp_model_cls", [PiecewiseLinearCobbDouglasGdpModel])
    if "personal_bankruptcy_model_cls" in metafunc.fixturenames:
        metafunc.parametrize("personal_bankruptcy_model_cls", [PersonalBankruptcyModel])
    if "corporate_bankruptcy_model_cls" in metafunc.fixturenames:
        metafunc.parametrize(
            "corporate_bankruptcy_model_cls",
            [NaiveCorporateBankruptcyModel, CorporateBankruptcyModel],
        )
    if "utilisation" in metafunc.fixturenames:
        metafunc.parametrize("utilisation", ALL_UTILISATIONS)


class TestClass:
    def test_interface(
        self,
        gdp_model_cls,
        personal_bankruptcy_model_cls,
        corporate_bankruptcy_model_cls,
        utilisation,
    ):
        reader = Reader(DATA_PATH)
        state = state_from_utilisation(UTILISATION_NO_COVID_NO_LOCKDOWN)
        econ_model = Economics(
            gdp_model_cls(),
            corporate_bankruptcy_model_cls(),
            personal_bankruptcy_model_cls(),
        )
        econ_model.load(reader)
        econ_model.simulate(state)
        new_state = advance_state(state, utilisation)
        econ_model.simulate(new_state)
        # Factor of 1.1 is because of the GDP backbone model
        assert (
            0
            <= sum(new_state.gdp_state.gdp.values())
            <= new_state.gdp_state.max_gdp * 1.1
        )
        for (
            _business_size,
            mapping,
        ) in new_state.corporate_state.proportion_solvent.items():
            for sector, solvent in mapping.items():
                assert 0 <= solvent <= 1
        for region in Region:
            assert 0 <= new_state.personal_state.personal_bankruptcy[region] <= 1
