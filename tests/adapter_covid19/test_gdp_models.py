"""
Basic example testing the adaptER-covid19 GDP models
"""

import sys

import numpy as np
import pandas as pd

from adapter_covid19.data_structures import SimulateState
from adapter_covid19.datasources import Reader
from adapter_covid19.enums import M, PrimaryInput
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

    def test_optimiser(self):
        reader = Reader(DATA_PATH)
        model = PiecewiseLinearCobbDouglasGdpModel()
        model.load(reader)
        setup = model.setup
        state = SimulateState(time=0, lockdown=False, utilisations=MAX_UTILISATIONS)

        def default_soln(v):
            # print(v)
            if v[0] == "q":
                return setup.q_iot[v[1]]
            elif v[0] == "d":
                return setup.dtilde_iot.loc[v[1], v[2]]
            elif v[0] == "x" or v[0] == "xtilde":
                return setup.xtilde_iot.loc[v[1], v[2]]
            elif v[0] == "y":
                return setup.ytilde_total_iot[v[1]]

        default_solution = [default_soln(v) for v in setup.variables]

        assert all([v is not None for v in default_solution])

        model.simulate(state)

        assert len(model.results.primary_inputs) == 1
        assert len(model.results.final_uses) == 1
        assert len(model.results.compensation_received) == 1
        assert len(model.results.compensation_paid) == 1
        assert len(model.results.compensation_subsidy) == 1
        assert len(model.results.max_primary_inputs) > 1
        assert len(model.results.max_final_uses) > 1
        assert len(model.results.max_compensation_paid) > 1
        assert len(model.results.max_compensation_received) > 1
        assert len(model.results.max_compensation_subsidy) > 1

        res = state.gdp_state._optimise_result
        x = res.x
        _df = pd.DataFrame(
            [x, default_solution], columns=setup.variables, index=["found", "default"]
        ).T
        _df["diff"] = _df["found"] - _df["default"]
        _df["err %"] = _df["diff"] / _df["default"]
        _df["var type"] = [v[0] for v in _df.index]
        _df[_df["var type"] == "xtilde"].sort_values("err %", ascending=False)

        # found and default final uses should be almost equal when aggregated
        np.testing.assert_allclose(
            _df[_df["var type"] == "y"]["found"].sum(),
            setup.ytilde_total_iot.sum(),
            rtol=1e-5,
            atol=0,
        )

        # objective function equals sum of default final uses minus imports
        np.testing.assert_allclose(
            -res.fun,
            setup.ytilde_total_iot.sum()
            - setup.xtilde_iot.loc[M.I].sum()
            - setup.o_iot.loc[PrimaryInput.TAXES_PRODUCTS].sum(),
            rtol=1e-5,
            atol=0,
        )

        # objective function equals sum of value adds
        np.testing.assert_allclose(
            -res.fun,
            setup.xtilde_iot.loc[M.K].sum()
            + setup.xtilde_iot.loc[M.L].sum()
            + setup.o_iot.loc[PrimaryInput.TAXES_PRODUCTION].sum(),
            rtol=1e-5,
            atol=0,
        )

        # found solution matches default solution for each variable
        np.testing.assert_allclose(_df["found"], _df["default"], rtol=1e-7, atol=1e0)
