import inspect
import itertools
import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

try:
    from tqdm import tqdm
except:

    def tqdm(x):
        return x


from adapter_covid19.datasources import Reader
from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel
from adapter_covid19.economics import Economics
from adapter_covid19 import gdp as gdp_models
from adapter_covid19.personal_insolvency import PersonalBankruptcyModel
from adapter_covid19.enums import Region, Sector, Age
from adapter_covid19.scenarios import Scenario


def lockdown_then_unlock_no_corona(
    data_path: Optional[str] = None,
    lockdown_on: int = 5,
    lockdown_off: int = 30,
    furlough_on: Optional[int] = 5,
    furlough_off: Optional[int] = 30,
    new_spending_day: int = 5,
    ccff_day: int = 5,
    loan_guarantee_day: int = 5,
    end_time: int = 50,
    gdp_model: str = "PiecewiseLinearCobbDouglasGdpModel",
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
    gdp_model_cls = gdp_models.__dict__[gdp_model]
    assert not inspect.isabstract(gdp_model_cls) and issubclass(
        gdp_model_cls, gdp_models.BaseGdpModel
    ), gdp_model
    gdp_model = gdp_model_cls(**init_args.gdp_kwargs)
    cb_model = CorporateBankruptcyModel(**init_args.corporate_kwargs)
    pb_model = PersonalBankruptcyModel(**init_args.personal_kwargs)
    econ = Economics(gdp_model, cb_model, pb_model, **init_args.economics_kwargs)
    econ.load(reader)
    healthy = {key: 1.0 for key in itertools.product(Region, Sector, Age)}
    ill = {key: 0.0 for key in itertools.product(Region, Sector, Age)}
    for i in tqdm(range(end_time)):
        simulate_state = scenario.generate(
            i,
            lockdown=lockdown_on <= i < lockdown_off,
            healthy=healthy,
            ill=ill
        )
        econ.simulate(simulate_state)
    df = (
        pd.DataFrame(
            [econ.results.fraction_gdp_by_sector(i) for i in range(1, end_time)],
            index=range(1, end_time),
        )
        .T.sort_index()
        .T.cumsum(axis=1)
    )

    if show_plots:
        # Plot 1
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.fill_between(df.index, df.iloc[:, 0] * 0, df.iloc[:, 0], label=df.columns[0])
        for i in range(1, df.shape[1]):
            ax.fill_between(
                df.index, df.iloc[:, i - 1], df.iloc[:, i], label=df.columns[i]
            )
        ax.legend(ncol=2)

        # Plot 2
        df = pd.DataFrame(
            [
                econ.results.corporate_solvencies[i]
                for i in econ.results.corporate_solvencies
            ]
        )
        df.plot(figsize=(20, 10))

        # Plot 3
        pd.DataFrame(
            [
                {
                    r: econ.results.personal_bankruptcy[i][r].personal_bankruptcy
                    for r in Region
                }
                for i in econ.results.personal_bankruptcy
            ]
        ).plot(figsize=(20, 10))

    return econ


def run_multiple_scenarios(data_path: str = None, show_plots: bool = True):
    scenario_results = {}
    scenario_results["no furlough"] = lockdown_then_unlock_no_corona(
        data_path=data_path,
        end_time=50,
        furlough_on=None,
        furlough_off=None,
        new_spending_day=1000,
        ccff_day=1000,
        loan_guarantee_day=1000,
        gdp_model="PiecewiseLinearCobbDouglasGdpModel",
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
        gdp_model="PiecewiseLinearCobbDouglasGdpModel",
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
        gdp_model="PiecewiseLinearCobbDouglasGdpModel",
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
        gdp_model="PiecewiseLinearCobbDouglasGdpModel",
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
        gdp_model="PiecewiseLinearCobbDouglasGdpModel",
        show_plots=show_plots,
    )
    return scenario_results


def plot_scenarios(scenarios, end_time=50, skip_scenarios=None):
    skip_scenarios = [] if skip_scenarios is None else skip_scenarios
    end_time = 50
    _scenarios = {n: e for n, e in scenarios.items() if n not in skip_scenarios}
    fig, axes = plt.subplots(
        3, len(_scenarios), sharex=True, sharey=True, figsize=(20, 10)
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
                {
                    r: econ.results.personal_bankruptcy[i][r].personal_bankruptcy
                    for r in Region
                }
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
