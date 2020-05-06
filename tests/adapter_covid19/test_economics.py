"""
Basic example testing the adaptER-covid19 economics class
"""

import sys

from adapter_covid19.data_structures import SimulateState
from adapter_covid19.economics import Economics

from adapter_covid19.corporate_bankruptcy import (
    CorporateBankruptcyModel,
    NaiveCorporateBankruptcyModel,
)
from adapter_covid19.enums import Region

from adapter_covid19.personal_insolvency import PersonalBankruptcyModel

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
)

sys.path.append("src/adapter_covid19")


def pytest_generate_tests(metafunc):
    if "gdp_model_cls" in metafunc.fixturenames:
        metafunc.parametrize("gdp_model_cls", [LinearGdpModel, SupplyDemandGdpModel])
    if "personal_bankruptcy_model_cls" in metafunc.fixturenames:
        metafunc.parametrize("personal_bankruptcy_model_cls", [PersonalBankruptcyModel])
    if "corporate_bankruptcy_model_cls" in metafunc.fixturenames:
        metafunc.parametrize(
            "corporate_bankruptcy_model_cls",
            [NaiveCorporateBankruptcyModel, CorporateBankruptcyModel],
        )
    if "utilisations" in metafunc.fixturenames:
        metafunc.parametrize("utilisations", [MAX_UTILISATIONS, MIN_UTILISATIONS])
    if "lockdown" in metafunc.fixturenames:
        metafunc.parametrize("lockdown", [True, False])


class TestClass:
    def test_interface(
        self,
        gdp_model_cls,
        personal_bankruptcy_model_cls,
        corporate_bankruptcy_model_cls,
        lockdown,
        utilisations,
    ):
        if issubclass(
            corporate_bankruptcy_model_cls, CorporateBankruptcyModel
        ) and not issubclass(gdp_model_cls, PiecewiseLinearCobbDouglasGdpModel):
            # Corp Bank model only compatible with Cobb Douglas Model
            return
        reader = Reader(DATA_PATH)
        state_0 = SimulateState(time=0, lockdown=False, utilisations=utilisations)
        state_1 = SimulateState(
            time=1, lockdown=lockdown, utilisations=utilisations, previous=state_0
        )
        econ_model = Economics(
            gdp_model_cls(),
            corporate_bankruptcy_model_cls(),
            personal_bankruptcy_model_cls(),
        )
        econ_model.load(reader)
        econ_model.simulate(state_0)
        econ_model.simulate(state_1)
        # Factor of 1.1 is because of the GDP backbone model
        assert (
            0 <= sum(state_1.gdp_state.gdp.values()) <= state_1.gdp_state.max_gdp * 1.1
        )
        for discount in state_1.corporate_state.gdp_discount_factor.values():
            assert 0 <= discount <= 1
        for region in Region:
            assert 0 <= state_1.personal_state.personal_bankruptcy[region] <= 1
