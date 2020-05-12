from datetime import date
from typing import List

import numpy as np
import pandas as pd

from adapter_covid19.data_structures import SimulateState
from adapter_covid19.economics import Economics

GDP_REDUCTION_EST = {
    date(2020, 3, 31): 0.97,
    date(2020, 6, 30): 0.75,
    date(2020, 9, 30): 0.82,
    date(2020, 12, 31): 0.90, # whole year: 0.86
}

DEMAND_REDUCTION_EST = {
    date(2020, 6, 30): 0.60,
}


def get_quarterly_gdp_decline(
        starting_date: date, states: List[SimulateState],
) -> pd.DataFrame:
    s_gdp_decline_simu = pd.DataFrame(
        [state.gdp_state.fraction_gdp_by_sector() for state in states],
        index=pd.date_range(start=starting_date, periods=len(states)),
    ).sum(axis=1)

    return pd.concat(
        [
            s_gdp_decline_simu.resample("1Q").last().rename("GDP decline simulation"),
            pd.Series(GDP_REDUCTION_EST).rename("GDP decline estimates"),
        ],
        axis=1,
    )


def get_quarterly_demand_decline(
        starting_date: date, econ: Economics, states: List[SimulateState],
) -> pd.DataFrame:
    s_demand_decline_simu_by_sector = pd.DataFrame(
        [state.personal_state.demand_reduction for state in states],
        index=pd.date_range(start=starting_date, periods=len(states)),
    )

    expense_by_sector = np.array(
        list(econ.personal_model.expenses_by_expense_sector.values())
    )

    s_demand_decline_simu = pd.Series(
        np.ndarray.flatten(
            np.dot(
                1 - s_demand_decline_simu_by_sector.values,
                expense_by_sector.reshape(-1, 1),
            )
        )
        / expense_by_sector.sum(),
        index=s_demand_decline_simu_by_sector.index,
    )

    return pd.concat(
        [
            s_demand_decline_simu.resample("1Q")
                .last()
                .rename("Demand decline simulation"),
            pd.Series(DEMAND_REDUCTION_EST).rename("Demand decline estimates"),
        ],
        axis=1,
    )
