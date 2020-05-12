import dataclasses
import itertools
import logging
import os
from typing import Optional, Mapping, Tuple, List, Dict

import matplotlib.pyplot as plt
import pandas as pd

from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel
from adapter_covid19.data_structures import Scenario, SimulateState
from adapter_covid19.datasources import Reader
from adapter_covid19.economics import Economics
from adapter_covid19.enums import (
    Region,
    Sector,
    Age,
    BusinessSize,
    WorkerState,
)
from adapter_covid19.gdp import PiecewiseLinearCobbDouglasGdpModel
from adapter_covid19.personal_insolvency import PersonalBankruptcyModel

try:
    from tqdm import tqdm
except:

    def tqdm(x):
        return x


logger = logging.getLogger(__file__)


class Simulator:
    def __init__(
        self, data_path: Optional[str] = None,
    ):
        if data_path is None:
            data_path = os.path.join(
                os.path.dirname(__file__), "../../tests/adapter_covid19/data"
            )
        self.reader = Reader(data_path)

    def simulate(
        self,
        scenario: Scenario,
        show_plots: bool = True,
        figsize: Tuple = (20, 60),
        scenario_name: str = "",
    ) -> Tuple[Economics, List[SimulateState]]:
        if not scenario.is_loaded:
            scenario.load(self.reader)

        model_params = scenario.model_params
        gdp_model = PiecewiseLinearCobbDouglasGdpModel(**model_params.gdp_params)
        cb_model = CorporateBankruptcyModel(**model_params.corporate_params)
        pb_model = PersonalBankruptcyModel(**model_params.personal_params)
        econ = Economics(gdp_model, cb_model, pb_model, **model_params.economics_params)
        econ.load(self.reader)
        dead = {key: 0.0 for key in itertools.product(Region, Sector, Age)}
        ill = {key: 0.0 for key in itertools.product(Region, Sector, Age)}
        states = []
        for i in tqdm(range(scenario.simulation_end_time)):
            simulate_state = scenario.generate(
                time=i,
                dead=dead,
                ill=ill,
                lockdown=scenario.lockdown_start_time <= i < scenario.lockdown_end_time,
                furlough=scenario.furlough_start_time <= i < scenario.furlough_end_time,
                reader=self.reader,
            )
            econ.simulate(simulate_state)
            states.append(simulate_state)

        if show_plots:
            fig, axes = plt.subplots(7, 1, sharex="col", sharey="row", figsize=figsize)
            plot_one_scenario(
                states, scenario.simulation_end_time, axes, title_prefix=scenario_name
            )

        return econ, states

    def simulate_multi(
        self,
        scenarios: Mapping[str, Scenario],
        show_plots: bool = True,
        figsize: Tuple = (20, 60),
    ) -> Dict[str, Tuple[Economics, List[SimulateState]]]:
        result = {}
        nl = "\n"
        for scenario_name, scenario in scenarios.items():
            logger.info(
                f"""
##################################################
# Scenario name: {scenario_name}
# Scenario details:
{nl.join([str(k) + ':' + str(v) for k, v in dataclasses.asdict(scenario).items()])}
##################################################"""
            )
            econ, states = self.simulate(
                scenario, show_plots, figsize, scenario_name=scenario_name + " ",
            )

            result[scenario_name] = econ, states

        return result


def plot_one_scenario(
    states, end_time, axes, title_prefix="", legend=False, return_dfs=False
):
    dfs = []

    # Plot 1a - Total GDP
    logger.debug("Plotting chart 1a")
    ax = axes[0]
    df = (
        pd.DataFrame(
            [states[i].gdp_state.fraction_gdp_by_sector() for i in range(1, end_time)],
            index=range(1, end_time),
        )
        .T.sort_index()
        .T.sum(axis=1)
    )
    df.plot(ax=ax)
    ax.legend(ncol=2)
    ax.set_title(title_prefix + "Total GDP")
    dfs.append(df)

    # Plot 1b - GDP Composition
    logger.debug("Plotting chart 1b")
    ax = axes[1]
    df = (
        pd.DataFrame(
            [states[i].gdp_state.fraction_gdp_by_sector() for i in range(1, end_time)],
            index=range(1, end_time),
        )
        .T.sort_index().T
    ).clip_lower(0.0)
    df = df.div(df.sum(axis=1),axis=0)
    df.plot.area(stacked=True,ax=ax)
    ax.legend(ncol=2)
    ax.set_title(title_prefix + "GDP Composition")
    dfs.append(df)

    # Plot 2a - Corporate Solvencies - Large Cap
    logger.debug("Plotting chart 2a")
    df = pd.DataFrame(
        [
            states[i].corporate_state.proportion_solvent[BusinessSize.large]
            for i in range(1, end_time)
        ]
    )
    df.plot(title=title_prefix + "Corporate Solvencies - Large Cap", ax=axes[2])
    dfs.append(df)

    # Plot 2b - Corporate Solvencies - SME
    logger.debug("Plotting chart 2b")
    df = pd.DataFrame(
        [
            states[i].corporate_state.proportion_solvent[BusinessSize.sme]
            for i in range(1, end_time)
        ]
    )
    df.plot(title=title_prefix + "Corporate Solvencies - SME", ax=axes[3])
    dfs.append(df)

    # Plot 3a - Personal Insolvencies
    logger.debug("Plotting chart 3a")
    df = pd.DataFrame(
        [states[i].personal_state.personal_bankruptcy for i in range(1, end_time)]
    )
    df.plot(title=title_prefix + "Personal Insolvencies", ax=axes[4])
    dfs.append(df)

    # Plot 3b - Household Expenditure
    logger.debug("Plotting chart 3b")
    df = pd.DataFrame(
        [states[i].personal_state.demand_reduction for i in range(1, end_time)]
    )
    df.plot(title=title_prefix + "Household Expenditure Reduction", ax=axes[5])
    dfs.append(df)

    # Plot 4 - Unemployment
    logger.debug("Plotting chart 4")

    def unemployment_from_lambdas(d):
        return (
            d[WorkerState.ILL_UNEMPLOYED]
            + d[WorkerState.HEALTHY_UNEMPLOYED]
            + d[WorkerState.ILL_FURLOUGHED]
            + d[WorkerState.HEALTHY_FURLOUGHED]
        ) / (1 - d[WorkerState.DEAD])

    df = pd.DataFrame(
        [
            {
                s: unemployment_from_lambdas(states[i].utilisations[s])
                for s in Sector
                if s != Sector.T_HOUSEHOLD
            }
            for i in range(1, end_time)
        ]
    )
    df.plot(title=title_prefix + "Unemployment", ax=axes[6])
    dfs.append(df)

    for ax in axes:
        if legend:
            ax.legend(ncol=2)
        else:
            ax.get_legend().remove()

    plt.tight_layout()

    if return_dfs:
        return dfs
