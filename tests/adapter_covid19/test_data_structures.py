"""
Basic example testing the adaptER-covid19 data structures
"""

import sys

import numpy as np
import pytest

from adapter_covid19.data_structures import Utilisation

sys.path.append("src/adapter_covid19")


utilisation_list = [
    Utilisation(
        p_dead=0.0001,
        p_ill_wfo=0.01,
        p_ill_wfh=0.01,
        p_ill_furloughed=0.01,
        p_ill_unemployed=0.01,
        p_wfh=0.7,
        p_furloughed=0.8,
        p_not_employed=0.5,
    )
]


class TestClass:
    @pytest.mark.parametrize("utilisation", utilisation_list)
    def test_utilisation(self, utilisation):
        lambdas = utilisation.to_lambdas()
        utilisation2 = Utilisation.from_lambdas(lambdas)
        assert utilisation == utilisation2
        lambdas2 = utilisation2.to_lambdas()
        assert not lambdas.keys() ^ lambdas2.keys()
        for key in lambdas:
            assert np.isclose(lambdas[key], lambdas2[key])
