"""
Basic example testing the adaptER-covid19 economics class
"""

import sys

from adapter_covid19.economics import Economics

from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel

from adapter_covid19.personal_insolvency import PersonalBankruptcyModel

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
    if "personal_bankruptcy_model_cls" in metafunc.fixturenames:
        metafunc.parametrize("personal_bankruptcy_model_cls", [PersonalBankruptcyModel])
    if "corporate_bankruptcy_model_cls" in metafunc.fixturenames:
        metafunc.parametrize(
            "corporate_bankruptcy_model_cls", [CorporateBankruptcyModel]
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
        reader = Reader(DATA_PATH)
        econ_model = Economics(
            gdp_model_cls(),
            corporate_bankruptcy_model_cls(),
            personal_bankruptcy_model_cls(),
        )
        econ_model.load(reader)
        try:
            # Simulation should fail unless we start outside of lockdown
            econ_model.simulate(time=0, lockdown=lockdown, utilisations=utilisations)
        except ValueError:
            if not lockdown:
                raise
        else:
            if lockdown:
                raise ValueError()
            assert 0 <= sum(econ_model.results.fraction_gdp_by_sector(0).values()) <= 1
