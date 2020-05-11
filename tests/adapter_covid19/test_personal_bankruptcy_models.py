"""
Basic example testing the adaptER-covid19 personal bankruptcy models
"""

import sys

from adapter_covid19.enums import Region

from adapter_covid19.datasources import Reader
from adapter_covid19.personal_insolvency import PersonalBankruptcyModel
from tests.adapter_covid19.utilities import (
    DATA_PATH,
    MAX_SECTOR_UTILISATIONS,
    MIN_SECTOR_UTILISATIONS,
)

sys.path.append("src/adapter_covid19")


def pytest_generate_tests(metafunc):
    if "personal_bankruptcy_model_cls" in metafunc.fixturenames:
        metafunc.parametrize("personal_bankruptcy_model_cls", [PersonalBankruptcyModel])
    if "gdp_discount" in metafunc.fixturenames:
        metafunc.parametrize(
            "gdp_discount", [MAX_SECTOR_UTILISATIONS, MIN_SECTOR_UTILISATIONS]
        )
    if "lockdown" in metafunc.fixturenames:
        metafunc.parametrize("lockdown", [True, False])


class TestClass:
    def test_interface(self, personal_bankruptcy_model_cls, lockdown, gdp_discount):
        reader = Reader(DATA_PATH)
        pb_model = personal_bankruptcy_model_cls()
        pb_model.load(reader)
        pb_model.simulate(
            time=0,
            lockdown=lockdown,
            corporate_bankruptcy={k: 1 - v for k, v in gdp_discount.items()},
        )
        for region in Region:
            assert 0 <= pb_model.results[0][region].personal_bankruptcy <= 1
