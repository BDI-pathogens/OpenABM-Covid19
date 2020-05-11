"""
Basic example testing the adaptER-covid19 corporate bankruptcy models
"""

import sys

from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel
from adapter_covid19.datasources import Reader
from tests.adapter_covid19.utilities import (
    DATA_PATH,
    MAX_SECTOR_UTILISATIONS,
    MIN_SECTOR_UTILISATIONS,
)

sys.path.append("src/adapter_covid19")


def pytest_generate_tests(metafunc):
    if "corporate_bankruptcy_model_cls" in metafunc.fixturenames:
        metafunc.parametrize(
            "corporate_bankruptcy_model_cls", [CorporateBankruptcyModel]
        )
    if "gdp_discount" in metafunc.fixturenames:
        metafunc.parametrize(
            "gdp_discount", [MAX_SECTOR_UTILISATIONS, MIN_SECTOR_UTILISATIONS]
        )


class TestClass:
    def test_interface(self, corporate_bankruptcy_model_cls, gdp_discount):
        reader = Reader(DATA_PATH)
        cb_model = corporate_bankruptcy_model_cls()
        cb_model.load(reader)
        discounts = cb_model.gdp_discount_factor(
            days_since_lockdown=1, gdp_discount=gdp_discount,
        )
        for discount in discounts.values():
            assert 0 <= discount <= 1
