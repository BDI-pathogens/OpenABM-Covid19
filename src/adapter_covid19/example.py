import itertools
import os
from typing import Optional
import logging
import pickle

import matplotlib.pyplot as plt
import pandas as pd

try:
    from tqdm import tqdm
except:

    def tqdm(x):
        return x


from adapter_covid19.datasources import Reader
from adapter_covid19.data_structures import Scenario
from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel
from adapter_covid19.gdp import PiecewiseLinearCobbDouglasGdpModel
from adapter_covid19.economics import Economics
from adapter_covid19.personal_insolvency import PersonalBankruptcyModel
from adapter_covid19.enums import (
    Region,
    Sector,
    Age,
    BusinessSize,
    WorkerState,
)
from adapter_covid19.scenarios import *
from adapter_covid19.simulator import Simulator

logger = logging.getLogger(__file__)


def lockdown_then_unlock_no_corona(
    data_path: Optional[str] = None,
    lockdown_on: int = 5,
    lockdown_off: int = 30,
    furlough_on: int = 5,
    furlough_off: int = 30,
    new_spending_day: int = 5,
    ccff_day: int = 5,
    loan_guarantee_day: int = 5,
    end_time: int = 50,
    show_plots: bool = True,
):
    """
    Lockdown at t=5 days, then release lockdown at t=30 days.

    :param data_path:
    :param lockdown_on:
    :param lockdown_off:
    :param end_time:
    :param gdp_model:
    :return:
    """
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__), "../../tests/adapter_covid19/data"
        )
    reader = Reader(data_path)
    scenario = Scenario(
        lockdown_start_time=lockdown_on,
        lockdown_end_time=lockdown_off,
        furlough_start_time=furlough_on,
        furlough_end_time=furlough_off,
        simulation_end_time=end_time,
        new_spending_day=new_spending_day,
        ccff_day=ccff_day,
        loan_guarantee_day=loan_guarantee_day,
        model_params=BASIC_MODEL_PARAMS,
    )
    scenario.load(reader)
    gdp_model = PiecewiseLinearCobbDouglasGdpModel(**scenario.model_params.gdp_params)
    cb_model = CorporateBankruptcyModel(**scenario.model_params.corporate_params)
    pb_model = PersonalBankruptcyModel(**scenario.model_params.personal_params)
    econ = Economics(
        gdp_model, cb_model, pb_model, **scenario.model_params.economics_params
    )
    econ.load(reader)
    dead = {key: 0.0 for key in itertools.product(Region, Sector, Age)}
    ill = {key: 0.0 for key in itertools.product(Region, Sector, Age)}
    states = []
    for i in tqdm(range(end_time)):
        simulate_state = scenario.generate(
            time=i,
            dead=dead,
            ill=ill,
            lockdown=lockdown_on <= i < lockdown_off,
            furlough=furlough_on <= i < furlough_off,
            reader=reader,
        )
        econ.simulate(simulate_state)
        states.append(simulate_state)

    if show_plots:
        fig, axes = plt.subplots(6, 1, sharex="col", sharey="row", figsize=(20, 60))
        plot_one_scenario(states, end_time, axes)

    return econ, states


def plot_one_scenario(states, end_time, axes, legend=False):
    # Plot 1 - GDP
    ax = axes[0]
    df = (
        pd.DataFrame(
            [states[i].gdp_state.fraction_gdp_by_sector() for i in range(1, end_time)],
            index=range(1, end_time),
        )
        .T.sort_index()
        .T.cumsum(axis=1)
    )
    ax.fill_between(df.index, df.iloc[:, 0] * 0, df.iloc[:, 0], label=df.columns[0])
    for i in range(1, df.shape[1]):
        ax.fill_between(df.index, df.iloc[:, i - 1], df.iloc[:, i], label=df.columns[i])
    ax.legend(ncol=2)
    ax.set_title("GDP")

    # Plot 2a - Corporate Solvencies - Large Cap
    df = pd.DataFrame(
        [
            states[i].corporate_state.proportion_solvent[BusinessSize.large]
            for i in range(1, end_time)
        ]
    )
    df.plot(title="Corporate Solvencies - Large Cap", ax=axes[1])

    # Plot 2b - Corporate Solvencies - SME
    df = pd.DataFrame(
        [
            states[i].corporate_state.proportion_solvent[BusinessSize.sme]
            for i in range(1, end_time)
        ]
    )
    df.plot(title="Corporate Solvencies - SME", ax=axes[2])

    # Plot 3a - Personal Insolvencies
    pd.DataFrame(
        [states[i].personal_state.personal_bankruptcy for i in range(1, end_time)]
    ).plot(title="Personal Insolvencies", ax=axes[3])

    # Plot 3b - Household Expenditure
    pd.DataFrame(
        [states[i].personal_state.demand_reduction for i in range(1, end_time)]
    ).plot(title="Household Expenditure Reduction", ax=axes[4])

    # Plot 4 - Unemployment
    def unemployment_from_lambdas(d):
        return (
            d[WorkerState.ILL_UNEMPLOYED]
            + d[WorkerState.HEALTHY_UNEMPLOYED]
            + d[WorkerState.ILL_FURLOUGHED]
            + d[WorkerState.HEALTHY_FURLOUGHED]
        ) / (1 - d[WorkerState.DEAD])

    pd.DataFrame(
        [
            {
                s: unemployment_from_lambdas(states[i].utilisations[s])
                for s in Sector
                if s != Sector.T_HOUSEHOLD
            }
            for i in range(1, end_time)
        ]
    ).plot(title="Unemployment", ax=axes[5])

    for ax in axes:
        if legend:
            ax.legend(ncol=2)
        else:
            ax.get_legend().remove()

    plt.tight_layout()


def plot_scenarios(scenarios, end_time=50):
    fig, axes = plt.subplots(
        6, len(scenarios), sharex="col", sharey="row", figsize=(20, 60 / len(scenarios))
    )
    for idx, (name, (econ, states)) in enumerate(scenarios.items()):
        axs = [row[idx] for row in axes]
        plot_one_scenario(states, end_time, axs)
    for ax, name in zip(axes[0], scenarios.values()):
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


def run_multiple_scenarios(data_path: str = None, show_plots: bool = True):
    scenario_results = {}
    scenario_results["no furlough"] = lockdown_then_unlock_no_corona(
        data_path=data_path,
        end_time=50,
        furlough_on=1000,
        furlough_off=1000,
        new_spending_day=1000,
        ccff_day=1000,
        loan_guarantee_day=1000,
        show_plots=show_plots,
    )
    scenario_results["furlough"] = lockdown_then_unlock_no_corona(
        data_path=data_path,
        end_time=50,
        furlough_on=5,
        furlough_off=30,
        new_spending_day=1000,
        ccff_day=1000,
        loan_guarantee_day=1000,
        show_plots=show_plots,
    )
    scenario_results[
        "furlough and corp support later"
    ] = lockdown_then_unlock_no_corona(
        data_path=data_path,
        end_time=50,
        furlough_on=5,
        furlough_off=30,
        new_spending_day=15,
        ccff_day=15,
        loan_guarantee_day=15,
        show_plots=show_plots,
    )
    # scenario_results["furlough and corp support"] = lockdown_then_unlock_no_corona(
    #     data_path=data_path,
    #     end_time=50,
    #     furlough_on=5,
    #     furlough_off=30,
    #     new_spending_day=5,
    #     ccff_day=5,
    #     loan_guarantee_day=5,
    #     show_plots=show_plots,
    # )
    # scenario_results[
    #     "furlough and corp spending only"
    # ] = lockdown_then_unlock_no_corona(
    #     data_path=data_path,
    #     end_time=50,
    #     furlough_on=5,
    #     furlough_off=30,
    #     new_spending_day=5,
    #     ccff_day=1000,
    #     loan_guarantee_day=1000,
    #     show_plots=show_plots,
    # )
    return scenario_results


if __name__ == "__main__":
    # uses simulator.py instead of the above logic
    # the above is deprecated

    logging.basicConfig(level=logging.INFO)

    import sys

    if len(sys.argv) > 1:
        scenario_names = sys.argv[1:]
    else:
        scenario_names = SCENARIOS.keys()

    unknown_scenarios = [n for n in scenario_names if n not in SCENARIOS.keys()]
    if len(unknown_scenarios) > 0:
        logger.error(f"Unkown scenarios! {unknown_scenarios}")
        exit(1)

    s = Simulator()

    for scenario_name in scenario_names:
        logger.info(f"Running scenario {scenario_name}")
        scenario = SCENARIOS[scenario_name]
        result = s.simulate(
            scenario=scenario, show_plots=False, scenario_name=scenario_name
        )
        file_name = f"scenario_{scenario_name}.pkl"
        logger.info(f"Writing file {file_name}")
        with open(file_name, "wb") as f:
            pickle.dump((scenario_name, scenario, result), f)
        logger.info(f"Finished writing")
