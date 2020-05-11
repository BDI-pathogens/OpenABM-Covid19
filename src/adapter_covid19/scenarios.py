from __future__ import annotations

from adapter_covid19.data_structures import Scenario, ModelParams

BASIC_MODEL_PARAMS = ModelParams(
    economics_params={},
    gdp_params={},
    personal_params={
        "default_th": 300,
        "max_earning_furloughed": 30_000,
        "alpha": 5,
        "beta": 20,
    },
    corporate_params={"beta": 1.4, "large_cap_cash_surplus_months": 6,},
)

BASIC_SCENARIO = Scenario(
    lockdown_recovery_time=1,
    lockdown_start_time=5,
    lockdown_end_time=30,
    furlough_start_time=5,
    furlough_end_time=30,
    simulation_end_time=50,
    new_spending_day=5,
    ccff_day=5,
    loan_guarantee_day=5,
    model_params=BASIC_MODEL_PARAMS,
)
