"""
Basic example testing the adaptER-covid19 GDP models
"""

import sys

import numpy as np
import pandas as pd

from adapter_covid19.corporate_bankruptcy import NaiveCorporateBankruptcyModel
from adapter_covid19.data_structures import PersonalState
from adapter_covid19.datasources import Reader
from adapter_covid19.enums import (
    M,
    Sector,
    LabourState,
    PrimaryInput,
)
from adapter_covid19.gdp import PiecewiseLinearCobbDouglasGdpModel
from tests.adapter_covid19.utilities import (
    DATA_PATH,
    ALL_UTILISATIONS,
    state_from_utilisation,
    UTILISATION_NO_COVID_NO_LOCKDOWN,
    advance_state,
)

DUMMY_PERSONAL_STATE = PersonalState(
    time=0,
    spot_earning={},
    spot_expense={},
    spot_expense_by_sector={},
    delta_balance={},
    balance={},
    credit_mean={},
    credit_std={},
    personal_bankruptcy={},
    demand_reduction={s: 1 for s in Sector},
)

sys.path.append("src/adapter_covid19")


def pytest_generate_tests(metafunc):
    if "gdp_model_cls" in metafunc.fixturenames:
        metafunc.parametrize(
            "gdp_model_cls", [PiecewiseLinearCobbDouglasGdpModel],
        )
    if "utilisation" in metafunc.fixturenames:
        metafunc.parametrize("utilisation", ALL_UTILISATIONS)


class TestClass:
    def test_interface(self, gdp_model_cls, utilisation):
        reader = Reader(DATA_PATH)
        state = state_from_utilisation(UTILISATION_NO_COVID_NO_LOCKDOWN)
        gdp_model = gdp_model_cls()
        gdp_model.load(reader)
        gdp_model.simulate(state)
        if issubclass(gdp_model_cls, PiecewiseLinearCobbDouglasGdpModel):
            # This is required because we need a `capital` param for the C-D model
            # TODO: make cleaner
            NaiveCorporateBankruptcyModel().simulate(state)
        new_state = advance_state(state, utilisation)
        new_state.previous.personal_state = DUMMY_PERSONAL_STATE
        gdp_model.simulate(new_state)
        # Factor of 1.1 is because of the GDP backbone model
        assert (
            0
            <= sum(new_state.gdp_state.gdp.values())
            <= new_state.gdp_state.max_gdp * 1.1
        )

    def test_optimiser(self):
        reader = Reader(DATA_PATH)
        model = PiecewiseLinearCobbDouglasGdpModel()
        model.load(reader)
        setup = model.setup
        state = state_from_utilisation(UTILISATION_NO_COVID_NO_LOCKDOWN)

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
            elif v[0] == "lambda":
                if v[1] == LabourState.WORKING:
                    return 1.0
                elif v[1] == LabourState.WFH:
                    return 0.0
                elif v[1] == LabourState.ILL:
                    return 0.0

        default_solution = [default_soln(v) for v in setup.variables]

        assert all([v is not None for v in default_solution])

        model.simulate(state)

        assert len(state.gdp_state.primary_inputs) > 1
        assert len(state.gdp_state.final_uses) > 1
        assert len(state.gdp_state.compensation_received) > 1
        assert len(state.gdp_state.compensation_paid) > 1
        assert len(state.gdp_state.compensation_subsidy) > 1

        res = state.gdp_state._optimise_result
        x = res.x
        _df = pd.DataFrame(
            [x, default_solution], columns=setup.variables, index=["found", "default"]
        ).T
        _df["diff"] = _df["found"] - _df["default"]
        _df["err %"] = _df["diff"] / _df["default"]
        _df["var type"] = [v[0] for v in _df.index]
        _df["var index 1"] = [v[1] if len(v) >= 2 else np.nan for v in _df.index]
        _df["var index 2"] = [v[2] if len(v) >= 3 else np.nan for v in _df.index]

        # found and default final uses should be almost equal when aggregated
        np.testing.assert_allclose(
            _df[_df["var type"] == "y"]["found"].sum(),
            setup.ytilde_total_iot.sum(),
            rtol=1e-3,
            atol=0,
        )

        # found and default gdp should be almost equal
        np.testing.assert_allclose(
            setup.get_gdp(x).sum(), setup.max_gdp, rtol=1e-3, atol=0,
        )

        # objective fn part 1 (gdp) equals sum of default final uses minus imports
        np.testing.assert_allclose(
            np.sum(list(setup.gdp_per_sector.values()), axis=0).dot(res.x),
            setup.ytilde_total_iot.sum()
            - setup.xtilde_iot.loc[M.I].sum()
            - setup.o_iot.loc[PrimaryInput.TAXES_PRODUCTS].sum(),
            rtol=1e-3,
            atol=0,
        )

        # objective fn part 1 (gdp) equals sum of default operating surplus
        np.testing.assert_allclose(
            np.sum(list(setup.gdp_per_sector.values()), axis=0).dot(res.x),
            setup.xtilde_iot.loc[M.K].sum()
            + setup.xtilde_iot.loc[M.L].sum()
            + setup.o_iot.loc[PrimaryInput.TAXES_PRODUCTION].sum(),
            rtol=1e-5,
            atol=0,
        )

        # objective fn part 2 (surplus) equals consumption + net operating surplus
        np.testing.assert_allclose(
            np.sum(list(setup.surplus_per_sector.values()), axis=0).dot(res.x),
            setup.xtilde_iot.loc[M.K].sum(),
            rtol=1e-5,
            atol=0,
        )

        # objective fn part 1 + part 2 = objective fn
        np.testing.assert_allclose(
            (
                np.sum(list(setup.gdp_per_sector.values()), axis=0)
                + np.sum(list(setup.surplus_per_sector.values()), axis=0)
            ).dot(res.x),
            -res.fun,
            rtol=1e-5,
            atol=0,
        )

        # found solution matches default solution for each variable
        # note: the household sector has lots of zero values, making the optimization problem underconstrained.
        #       therefore, we ignore variables relating to the household sector in below checks
        __df = _df[
            ~(
                (_df["var index 1"] == Sector.T_HOUSEHOLD)
                | (_df["var index 2"] == Sector.T_HOUSEHOLD)
            )
        ]
        np.testing.assert_allclose(__df["found"], __df["default"], rtol=1e-4, atol=1e0)
        # all non-lambda variables are in currency, therefore large atol is acceptable
        np.testing.assert_allclose(
            __df[__df["var type"] != "lambda"]["found"],
            __df[__df["var type"] != "lambda"]["default"],
            rtol=1e-8,
            atol=2e0,
        )
        # all lambda variables are in [0,1] therefore atol has to be small
        np.testing.assert_allclose(
            __df[__df["var type"] == "lambda"]["found"],
            __df[__df["var type"] == "lambda"]["default"],
            rtol=1e-5,
            atol=1e-5,
        )
