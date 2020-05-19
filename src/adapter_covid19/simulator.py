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
    FinalUse,
    M,
)
from adapter_covid19.gdp import PiecewiseLinearCobbDouglasGdpModel
from adapter_covid19.personal_insolvency import PersonalBankruptcyModel

try:
    from tqdm import tqdm
except:

    def tqdm(x):
        return x


N_PLOTS = 14

CHART_NAMES = [
    "Deaths",
    "People Ill",
    "Total GDP",
    "GDP Composition",
    "Capital Stock",
    "Investment",
    "Corporate Solvencies - Large Cap",
    "Corporate Solvencies - SME",
    "Household Expenditure Reduction (Total)",
    "Household Expenditure Reduction by Sector",
    "Fear Factor",
    "Opportunity Gap",
    "Unemployed + Furloughed by Sector",
    "Unemployed vs Furloughed",
]

logger = logging.getLogger(__file__)


class Simulator:
    """Simulator for adaptER-covid19"""

    def __init__(
        self, data_path: Optional[str] = None,
    ):
        """
        Simulator for adaptER-covid19

        :param data_path: path to data
        :type data_path: Optional[str]
        """
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
        """
        Run simulation for a given scenario

        :param scenario: Scenario to run the simulation
        :type scenario: Scenario
        :param show_plots: Show the plots using matplotlib if set to be True
        :type show_plots: bool
        :param figsize: Size of the figure to be plotted. This parameter is only used when show_plots is set to be True
        :type figsize: Tuple
        :param scenario_name: Name of the scenario. Used in plotting
        :type scenario_name: str
        :return: Tuple[Economics, List[SimulateState]]
        """
        # Load data for the scenario object if not done yet
        if not scenario.is_loaded:
            scenario.load(self.reader)

        # Initialize various models given the model parameters specified in the Scenario
        model_params = scenario.model_params
        gdp_model = PiecewiseLinearCobbDouglasGdpModel(**model_params.gdp_params)
        cb_model = CorporateBankruptcyModel(**model_params.corporate_params)
        pb_model = PersonalBankruptcyModel(**model_params.personal_params)
        econ = Economics(gdp_model, cb_model, pb_model, **model_params.economics_params)

        # Load data for Economy
        econ.load(self.reader)

        states = []
        for i in tqdm(range(scenario.simulation_end_time)):
            # Load ill and dead ratio from scenario
            ill = scenario.get_ill_ratio_dict(i)
            dead = scenario.get_dead_ratio_dict(i)

            # Load quarantine information
            quarantine = scenario.get_quarantine_ratio_dict(i)

            # Initialize  a SimulateState object
            simulate_state = scenario.generate(
                time=i,
                dead=dead,
                ill=ill,
                quarantine=quarantine,
                lockdown=scenario.lockdown_start_time <= i < scenario.lockdown_end_time,
                furlough=scenario.furlough_start_time <= i < scenario.furlough_end_time,
                reader=self.reader,
            )

            # Pass SimulateState to Economy
            econ.simulate(simulate_state)

            # Keep a record of the current SimulateState
            states.append(simulate_state)

        # Plots
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
        """
        Run simulation for multiple given scenarios in sequence

        :param scenarios: Dictionary of Scenarios to run the simulation
        :type scenarios: Mapping[str, Scenario]
        :param show_plots: Show the plots using matplotlib if set to be True
        :type show_plots: bool
        :param figsize: Size of the figure to be plotted. This parameter is only used when show_plots is set to be True
        :type figsize: Tuple
        :return: Dict[str, Tuple[Economics, List[SimulateState]]]
        """
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


def summarize_one_scenario(econ, states, end_time, start_date=None):
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
            for i in range(end_time)
        ]
    ).sum(axis=1) / pd.Series(
        [states[i].gdp_state.max_workers for i in range(end_time)]
    )
    dfs["People Ill"] = df

    # Table 1a - Total GDP
    df = (
        pd.DataFrame(
            [states[i].gdp_state.fraction_gdp_by_sector() for i in range(end_time)],
        )
        .T.sort_index()
        .T.sum(axis=1)
    )
    dfs["Total GDP"] = df

    # Table 1b - GDP Composition
    df = (
        pd.DataFrame(
            [states[i].gdp_state.fraction_gdp_by_sector() for i in range(end_time)],
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
            for i in range(end_time)
        ],
    )
    dfs["Corporate Solvencies - Large Cap"] = df

    # Table 2b - Corporate Solvencies - SME
    df = pd.DataFrame(
        [
            states[i].corporate_state.proportion_solvent[BusinessSize.sme]
            for i in range(end_time)
        ],
    )
    dfs["Corporate Solvencies - SME"] = df

    # Table 2c - Capital Stock
    df = (
        pd.DataFrame(
            [
                states[i].corporate_state.capital_discount_factor
                for i in range(end_time)
            ],
        )
        .multiply(
            (
                econ.gdp_model.setup.xtilde_iot.loc[M.K]
                / econ.gdp_model.setup.xtilde_iot.loc[M.K].sum()
            ),
            axis=1,
        )
        .sum(axis=1)
    )
    dfs["Capital Stock"] = df

    # Table 2d - Investment
    df = pd.DataFrame([state.corporate_state.exhuberance_factor for state in states])
    dfs["Investment"] = df

    # Table 3a - Personal Insolvencies
    df = pd.DataFrame(
        [states[i].personal_state.personal_bankruptcy for i in range(end_time)],
    )
    dfs["Personal Insolvencies"] = df

    # Table 3b - Household Expenditure
    df = pd.DataFrame(
        [states[i].personal_state.demand_reduction for i in range(end_time)],
    )
    dfs["Household Expenditure Reduction by Sector"] = df
    dfs["Household Expenditure Reduction (Total)"] = (
        df.multiply(econ.gdp_model.setup.ytilde_iot[FinalUse.C]).sum(axis=1)
        / econ.gdp_model.setup.ytilde_iot[FinalUse.C].sum()
    )

    # Table 3c - Fear Factor
    df = pd.DataFrame([states[i].get_fear_factor() for i in range(len(states))])
    dfs["Fear Factor"] = df

    # Table 3d - Opportunity Gap
    df = (
        -pd.DataFrame(
            [state.gdp_state.final_use_shortfall_vs_demand for state in states]
        )
        + 1.0
    )
    dfs["Opportunity Gap"] = df

    # Table 4a - Unemployment & Furloughing (by Sector)
    # TODO: Leverage utilisations class for the below computations instead
    def not_employment_from_lambdas(d):
        return (
            d[WorkerState.ILL_UNEMPLOYED]
            + d[WorkerState.HEALTHY_UNEMPLOYED]
            + d[WorkerState.ILL_FURLOUGHED]
            + d[WorkerState.HEALTHY_FURLOUGHED]
        ) / (1 - d[WorkerState.DEAD])

    def unemployment_from_lambdas(d):
        return (d[WorkerState.ILL_UNEMPLOYED] + d[WorkerState.HEALTHY_UNEMPLOYED]) / (
            1 - d[WorkerState.DEAD]
        )

    def furloughing_from_lambdas(d):
        return (+d[WorkerState.ILL_FURLOUGHED] + d[WorkerState.HEALTHY_FURLOUGHED]) / (
            1 - d[WorkerState.DEAD]
        )

    df = pd.DataFrame(
        [
            {
                s: not_employment_from_lambdas(states[i].utilisations[s])
                for s in Sector
                if s != Sector.T_HOUSEHOLD
            }
            for i in range(end_time)
        ],
    )
    dfs["Unemployed + Furloughed by Sector"] = df

    # Table 4b - Unemployment & Furloughing (by Unemployed / Furloughed)
    max_workers_by_sector = {
        s: sum(
            econ.gdp_model.workers[r, s, a] for r, a in itertools.product(Region, Age)
        )
        for s in Sector
    }
    df_workers_alive = pd.DataFrame(
        [
            {
                s: (1 - states[i].utilisations[s][WorkerState.DEAD])
                * max_workers_by_sector[s]
                for s in Sector
            }
            for i in range(end_time)
        ],
    )

    df_u = pd.DataFrame(
        [
            {
                s: unemployment_from_lambdas(states[i].utilisations[s])
                * df_workers_alive.loc[i, s]
                for s in Sector
            }
            for i in range(end_time)
        ],
    )

    df_f = pd.DataFrame(
        [
            {
                s: furloughing_from_lambdas(states[i].utilisations[s])
                * df_workers_alive.loc[i, s]
                for s in Sector
            }
            for i in range(end_time)
        ],
    )
    df = pd.DataFrame(
        {
            "unemployed": df_u.sum(axis=1) / df_workers_alive.sum(axis=1),
            "furloughed": df_f.sum(axis=1) / df_workers_alive.sum(axis=1),
        }
    )
    dfs["Unemployed vs Furloughed"] = df

    if start_date is not None:
        for df in dfs.values():
            idx = pd.date_range(start_date, periods=len(df))
            df.index = idx

    return dfs


def metrics_one_scenario(dfs, scenario_name="Scenario"):
    _metric_dfs = []

    table_name = "Total GDP"
    _df = dfs[table_name].resample("1Q").mean().iloc[1:].to_frame(name=scenario_name)
    _df = _df.reset_index().rename(columns={"index": "Period"})
    _df["Metric"] = table_name
    _df["Period"] = _df["Period"].replace(
        to_replace={
            pd.to_datetime("2020-06-30"): "Q2",
            pd.to_datetime("2020-09-30"): "Q3",
        }
    )
    _df = _df.set_index(["Metric", "Period"])
    _metric_dfs.append(_df)

    table_name = "Household Expenditure Reduction (Total)"
    _df = dfs[table_name].resample("1Q").mean().iloc[1:].to_frame(name=scenario_name)
    _df = _df.reset_index().rename(columns={"index": "Period"})
    _df["Metric"] = table_name
    _df["Period"] = _df["Period"].replace(
        to_replace={
            pd.to_datetime("2020-06-30"): "Q2",
            pd.to_datetime("2020-09-30"): "Q3",
        }
    )
    _df = _df.set_index(["Metric", "Period"])
    _metric_dfs.append(_df)

    table_name = "Capital Stock"
    _df = dfs[table_name].resample("1Q").mean().iloc[1:].to_frame(name=scenario_name)
    _df = _df.reset_index().rename(columns={"index": "Period"})
    _df["Metric"] = table_name
    _df["Period"] = _df["Period"].replace(
        to_replace={
            pd.to_datetime("2020-06-30"): "Q2",
            pd.to_datetime("2020-09-30"): "Q3",
        }
    )
    _df = _df.set_index(["Metric", "Period"])
    _metric_dfs.append(_df)

    table_name = "Unemployed vs Furloughed"
    _mdf = dfs[table_name].resample("1Q").mean().iloc[1:]

    _df = _mdf["unemployed"].to_frame(name=scenario_name)
    _df = _df.reset_index().rename(columns={"index": "Period"})
    _df["Metric"] = "Unemployed"
    _df["Period"] = _df["Period"].replace(
        to_replace={
            pd.to_datetime("2020-06-30"): "Q2",
            pd.to_datetime("2020-09-30"): "Q3",
        }
    )
    _df = _df.set_index(["Metric", "Period"])
    _metric_dfs.append(_df)

    _df = _mdf["furloughed"].to_frame(name=scenario_name)
    _df = _df.reset_index().rename(columns={"index": "Period"})
    _df["Metric"] = "Furloughed"
    _df["Period"] = _df["Period"].replace(
        to_replace={
            pd.to_datetime("2020-06-30"): "Q2",
            pd.to_datetime("2020-09-30"): "Q3",
        }
    )
    _df = _df.set_index(["Metric", "Period"])
    _metric_dfs.append(_df)

    metrics_df = pd.concat(_metric_dfs)

    return metrics_df


def metrics_scenarios(scenarios):
    mdfs = [metrics_one_scenario(dfs, name) for name, dfs in scenarios.items()]
    return pd.concat(mdfs, axis=1)


def plot_one_scenario(dfs, axes, title_prefix="", legend=False, title=True):
    for i, chart_name in enumerate(CHART_NAMES):
        logger.debug(f"Plotting chart {chart_name}")
        df = dfs[chart_name]
        title = title_prefix + chart_name if title else ""
        if chart_name == "GDP Composition":
            df.plot.area(stacked=True, ax=axes[i], title=title)
        else:
            df.plot(ax=axes[i], title=title)

    for ax in axes:
        if legend:
            ax.legend(ncol=2)
        else:
            l = ax.get_legend()
            if l is not None:
                l.remove()

    plt.tight_layout()


def plot_scenarios(scenarios, end_time=50):
    n_charts = N_PLOTS
    fig, axes = plt.subplots(
        n_charts,
        len(scenarios),
        sharex="col",
        sharey="row",
        figsize=(5 * len(scenarios), 3 * n_charts),
    )
    for idx, (name, dfs) in enumerate(scenarios.items()):
        axs = [row[idx] for row in axes]
        plot_one_scenario(dfs, axs, title=False)
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
    for ax, chart_name in zip(axes[:, 0], CHART_NAMES):
        ax.annotate(
            chart_name,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
        )
