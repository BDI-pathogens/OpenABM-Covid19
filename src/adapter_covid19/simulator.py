import dataclasses
import itertools
import logging
import os
from typing import Optional, Mapping, Tuple, List, Dict
import matplotlib.pyplot as plt
import time
import pandas as pd

from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel
from adapter_covid19.data_structures import Scenario, SimulateState, ModelParams
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

        states = []
        for i in tqdm(range(scenario.simulation_end_time)):
            ill = scenario.get_ill_ratio_dict(i)
            dead = scenario.get_dead_ratio_dict(i)

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


def summarize_one_scenario(
    states, end_time,
):
    dfs = {}

    # Table 0a - Death
    # TODO: connect this to deaths from the epidemic simulation rather than utilisation
    df = pd.DataFrame(
        [
            {
                s: states[i].utilisations[s][WorkerState.DEAD]
                * states[i].gdp_state.workers_in_sector(s)
                for s in Sector
            }
            for i in range(1, end_time)
        ],
        index=range(1, end_time),
    ).sum(axis=1) / pd.Series(
        [states[i].gdp_state.max_workers for i in range(1, end_time)],
        index=range(1, end_time),
    )
    dfs["Deaths"] = df

    # Table 0b - Illness
    # TODO: connect this to deaths from the epidemic simulation rather than utilisation
    def illness_from_lambdas(d):
        return (
            d[WorkerState.ILL_UNEMPLOYED]
            + d[WorkerState.ILL_FURLOUGHED]
            + d[WorkerState.ILL_WFH]
            + d[WorkerState.ILL_WFO]
        )

    df = pd.DataFrame(
        [
            {
                s: illness_from_lambdas(states[i].utilisations[s])
                * states[i].gdp_state.workers_in_sector(s)
                for s in Sector
            }
            for i in range(1, end_time)
        ],
        index=range(1, end_time),
    ).sum(axis=1) / pd.Series(
        [states[i].gdp_state.max_workers for i in range(1, end_time)],
        index=range(1, end_time),
    )
    dfs["People Ill"] = df

    # Table 1a - Total GDP
    df = (
        pd.DataFrame(
            [states[i].gdp_state.fraction_gdp_by_sector() for i in range(1, end_time)],
            index=range(1, end_time),
        )
        .T.sort_index()
        .T.sum(axis=1)
    )
    dfs["Total GDP"] = df

    # Table 1b - GDP Composition
    df = (
        pd.DataFrame(
            [states[i].gdp_state.fraction_gdp_by_sector() for i in range(1, end_time)],
            index=range(1, end_time),
        )
        .T.sort_index()
        .T
    ).clip(lower=0.0)
    df = df.div(df.sum(axis=1), axis=0)
    dfs["GDP Composition"] = df

    # Table 2a - Corporate Solvencies - Large Cap
    df = pd.DataFrame(
        [
            states[i].corporate_state.proportion_solvent[BusinessSize.large]
            for i in range(1, end_time)
        ]
    )
    dfs["Corporate Solvencies - Large Cap"] = df

    # Table 2b - Corporate Solvencies - SME
    df = pd.DataFrame(
        [
            states[i].corporate_state.proportion_solvent[BusinessSize.sme]
            for i in range(1, end_time)
        ]
    )
    dfs["Corporate Solvencies - SME"] = df

    # Table 3a - Personal Insolvencies
    df = pd.DataFrame(
        [states[i].personal_state.personal_bankruptcy for i in range(1, end_time)]
    )
    dfs["Personal Insolvencies"] = df

    # Table 3b - Household Expenditure
    df = pd.DataFrame(
        [states[i].personal_state.demand_reduction for i in range(1, end_time)]
    )
    dfs["Household Expenditure Reduction"] = df

    # Table 4 - Unemployment & Furloughing
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
    dfs["Unemployment & Furloughing"] = df

    return dfs


def plot_one_scenario(dfs, axes, title_prefix="", legend=False):
    # Plot 0a - Deaths
    logger.debug("Plotting chart 0a")
    chart_name = "Deaths"
    df = dfs[chart_name]
    df.plot(ax=axes[0], title=title_prefix + chart_name)

    # Plot 0b - Illness
    logger.debug("Plotting chart 0b")
    chart_name = "People Ill"
    df = dfs[chart_name]
    df.plot(ax=axes[1], title=title_prefix + chart_name)

    # Plot 1a - Total GDP
    logger.debug("Plotting chart 1a")
    chart_name = "Total GDP"
    df = dfs[chart_name]
    df.plot(ax=axes[2], title=title_prefix + chart_name)

    # Plot 1b - GDP Composition
    logger.debug("Plotting chart 1b")
    chart_name = "GDP Composition"
    df = dfs[chart_name]
    df.plot.area(stacked=True, ax=axes[3], title=title_prefix + "GDP Composition")

    # Plot 2a - Corporate Solvencies - Large Cap
    logger.debug("Plotting chart 2a")
    chart_name = "Corporate Solvencies - Large Cap"
    df = dfs[chart_name]
    df.plot(title=title_prefix + chart_name, ax=axes[4])

    # Plot 2b - Corporate Solvencies - SME
    logger.debug("Plotting chart 2b")
    chart_name = "Corporate Solvencies - SME"
    df = dfs[chart_name]
    df.plot(title=title_prefix + chart_name, ax=axes[5])

    # Plot 3a - Personal Insolvencies
    logger.debug("Plotting chart 3a")
    chart_name = "Personal Insolvencies"
    df = dfs[chart_name]
    df.plot(title=title_prefix + chart_name, ax=axes[6])

    # Plot 3b - Household Expenditure
    logger.debug("Plotting chart 3b")
    chart_name = "Household Expenditure Reduction"
    df = dfs[chart_name]
    df.plot(title=title_prefix + chart_name, ax=axes[7])

    # Plot 4 - Unemployment & Furloughing
    logger.debug("Plotting chart 4")
    chart_name = "Unemployment & Furloughing"
    df = dfs[chart_name]
    df.plot(title=title_prefix + chart_name, ax=axes[8])

    for ax in axes:
        if legend:
            ax.legend(ncol=2)
        else:
            l = ax.get_legend()
            if l is not None:
                l.remove()

    plt.tight_layout()


def plot_scenarios(scenarios, end_time=50):
    fig, axes = plt.subplots(
        9,
        len(scenarios),
        sharex="col",
        sharey="row",
        figsize=(3.5 * len(scenarios), 2 * 9),
    )
    for idx, (name, dfs) in enumerate(scenarios.items()):
        axs = [row[idx] for row in axes]
        plot_one_scenario(dfs, axs)
    for ax, name in zip(axes[0], [k for k in scenarios.keys()]):
        ax.annotate(
            name,
            xy=(0.5, 1),
            xytext=(0, 5),
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
        )
