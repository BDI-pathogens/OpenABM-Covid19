import inspect
import itertools
import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from adapter_covid19.datasources import Reader
from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel
from adapter_covid19.economics import Economics
from adapter_covid19 import gdp as gdp_models
from adapter_covid19.personal_insolvency import PersonalBankruptcyModel
from adapter_covid19.enums import Region, Sector, Age, LabourState
from adapter_covid19.scenarios import Scenario


def lockdown_then_unlock_no_corona(
    data_path: str = Optional[str] = None,
    lockdown_on: int = 5,
    lockdown_off: int = 30,
    end_time: int = 50,
    gdp_model: str = "SupplyDemandGdpModel",
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
    gdp_model_cls = gdp_models.__dict__[gdp_model]
    assert not inspect.isabstract(gdp_model_cls) and issubclass(
        gdp_model_cls, gdp_models.BaseGdpModel
    ), gdp_model
    scenario = Scenario()
    scenario.load(reader)
    econ = Economics(
        gdp_model_cls(), CorporateBankruptcyModel(), PersonalBankruptcyModel()
    )
    econ.load(reader)
    healthy = {key: 1.0 for key in itertools.product(Region, Sector, Age)}
    ill = {key: 0.0 for key in itertools.product(Region, Sector, Age)}
    for i in range(end_time):
        if lockdown_on <= i < lockdown_off:
            utilisations = scenario.generate(i, lockdown=True, healthy=healthy, ill=ill)
            econ.simulate(i, lockdown=True, utilisations=utilisations)
        else:
            utilisations = scenario.generate(
                i, lockdown=False, healthy=healthy, ill=ill
            )
            econ.simulate(i, lockdown=False, utilisations=utilisations)
    df = (
        pd.DataFrame(
            [econ.results.fraction_gdp_by_sector(i) for i in range(1, end_time)],
            index=range(1, end_time),
        )
        .T.sort_index()
        .T.cumsum(axis=1)
    )

    # Plot 1
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.fill_between(df.index, df.iloc[:, 0] * 0, df.iloc[:, 0], label=df.columns[0])
    for i in range(1, df.shape[1]):
        ax.fill_between(df.index, df.iloc[:, i - 1], df.iloc[:, i], label=df.columns[i])
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        lockdown_then_unlock_no_corona(*sys.argv[1:])
    else:
        lockdown_then_unlock_no_corona()
