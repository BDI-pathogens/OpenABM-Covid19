import itertools
import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from adapter_covid19.gdp import PiecewiseLinearCobbDouglasGdpModel

try:
    from tqdm import tqdm
except:

    def tqdm(x):
        return x


from adapter_covid19.datasources import Reader
from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel
from adapter_covid19.economics import Economics
from adapter_covid19.personal_insolvency import PersonalBankruptcyModel
from adapter_covid19.enums import Region, Sector, Age, BusinessSize, WorkerState
from adapter_covid19.scenarios import Scenario


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
        furlough_start_time=furlough_on,
        furlough_end_time=furlough_off,
        new_spending_day=new_spending_day,
        ccff_day=ccff_day,
        loan_guarantee_day=loan_guarantee_day,
    )
    scenario.load(reader)
    init_args = scenario.initialise()
    gdp_model = PiecewiseLinearCobbDouglasGdpModel(**init_args.gdp_kwargs)
    cb_model = CorporateBankruptcyModel(**init_args.corporate_kwargs)
    pb_model = PersonalBankruptcyModel(**init_args.personal_kwargs)
    econ = Economics(gdp_model, cb_model, pb_model, **init_args.economics_kwargs)
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
        )
        econ.simulate(simulate_state)
        states.append(simulate_state)

    if show_plots:
        # Plot 1 - GDP
        df = (
            pd.DataFrame(
                [
                    states[i].gdp_state.fraction_gdp_by_sector()
                    for i in range(1, end_time)
                ],
                index=range(1, end_time),
            )
            .T.sort_index()
            .T.cumsum(axis=1)
        )
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.fill_between(df.index, df.iloc[:, 0] * 0, df.iloc[:, 0], label=df.columns[0])
        for i in range(1, df.shape[1]):
            ax.fill_between(
                df.index, df.iloc[:, i - 1], df.iloc[:, i], label=df.columns[i]
            )
        ax.legend(ncol=2)
        ax.set_title("GDP")

        # Plot 2a - Corporate Solvencies - Large Cap
        df = pd.DataFrame(
            [
                states[i].corporate_state.proportion_solvent[BusinessSize.large]
                for i in range(1, end_time)
            ]
        )
        df.plot(figsize=(20, 10), title="Corporate Solvencies - Large Cap")

        # Plot 2b - Corporate Solvencies - SME
        df = pd.DataFrame(
            [
                states[i].corporate_state.proportion_solvent[BusinessSize.sme]
                for i in range(1, end_time)
            ]
        )
        df.plot(figsize=(20, 10), title="Corporate Solvencies - SME")

        # Plot 3a - Personal Insolvencies
        pd.DataFrame(
            [states[i].personal_state.personal_bankruptcy for i in range(1, end_time)]
        ).plot(figsize=(20, 10), title="Personal Insolvencies")

        # Plot 3b - Household Expenditure
        pd.DataFrame(
            [states[i].personal_state.demand_reduction for i in range(1, end_time)]
        ).plot(figsize=(20, 10), title="Household Expenditure Reduction")

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
                }
                for i in range(1, end_time)
            ]
        ).plot(figsize=(20, 10), title="Unemployment")

    return econ, states


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
    scenario_results["furlough and corp support"] = lockdown_then_unlock_no_corona(
        data_path=data_path,
        end_time=50,
        furlough_on=5,
        furlough_off=30,
        new_spending_day=5,
        ccff_day=5,
        loan_guarantee_day=5,
        show_plots=show_plots,
    )
    scenario_results[
        "furlough and corp spending only"
    ] = lockdown_then_unlock_no_corona(
        data_path=data_path,
        end_time=50,
        furlough_on=5,
        furlough_off=30,
        new_spending_day=5,
        ccff_day=1000,
        loan_guarantee_day=1000,
        show_plots=show_plots,
    )
    return scenario_results


def plot_scenarios(scenarios, end_time=50, skip_scenarios=None):
    skip_scenarios = [] if skip_scenarios is None else skip_scenarios
    end_time = 50
    _scenarios = {n: e for n, e in scenarios.items() if n not in skip_scenarios}
    fig, axes = plt.subplots(
        4, len(_scenarios), sharex=True, sharey=True, figsize=(20, 12)
    )
    for idx, (name, econ) in enumerate(_scenarios.items()):
        # Plot 1
        ax = axes[0][idx]
        df = (
            pd.DataFrame(
                [econ.results.fraction_gdp_by_sector(i) for i in range(1, end_time)],
                index=range(1, end_time),
            )
            .T.sort_index()
            .T.cumsum(axis=1)
        )
        ax.fill_between(df.index, df.iloc[:, 0] * 0, df.iloc[:, 0], label=df.columns[0])
        for i in range(1, df.shape[1]):
            ax.fill_between(
                df.index, df.iloc[:, i - 1], df.iloc[:, i], label=df.columns[i]
            )
        # ax.legend(ncol=2)
        ax.legend().remove()
        ax.set_title(name)

        # Plot 2
        ax = axes[1][idx]
        df = pd.DataFrame(
            [
                econ.results.corporate_solvencies[i]
                for i in econ.results.corporate_solvencies
            ]
        )
        df.plot(ax=ax)
        ax.legend().remove()

        # Plot 3
        ax = axes[2][idx]
        pd.DataFrame(
            [
                econ.results.personal_bankruptcy[i].personal_bankruptcy
                for i in econ.results.personal_bankruptcy
            ]
        ).plot(ax=ax)
        ax.legend().remove()

        # Plot 4
        ax = axes[3][idx]
        pd.DataFrame(
            [
                econ.results.personal_bankruptcy[i].demand_reduction
                for i in econ.results.personal_bankruptcy
            ]
        ).plot(ax=ax)
        ax.legend().remove()
    plt.tight_layout()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run_multiple_scenarios(*sys.argv[1:])
    else:
        run_multiple_scenarios(show_plots=False)
        # plot_scenarios(scenarios)
