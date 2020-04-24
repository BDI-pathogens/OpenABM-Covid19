import itertools

import matplotlib.pyplot as plt
import pandas as pd

from adapter_covid19.datasources import Reader
from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel
from adapter_covid19.economics import Economics
from adapter_covid19.gdp import SupplyDemandGdpModel
from adapter_covid19.personal_insolvency import PersonalBankruptcyModel
from adapter_covid19.enums import Region, Sector, Age


def lockdown_then_unlock_no_corona(data_path: str = "data"):
    """
    Lockdown at t=5 days, then release lockdown at t=50 days.

    :param data_path:
    :return:
    """
    reader = Reader(data_path)
    econ = Economics(
        SupplyDemandGdpModel(), CorporateBankruptcyModel(), PersonalBankruptcyModel()
    )
    econ.load(reader)
    max_utilisations = {key: 1.0 for key in itertools.product(Region, Sector, Age)}
    min_utilisations = {key: 0.0 for key in itertools.product(Region, Sector, Age)}
    length = 100
    for i in range(length):
        if 5 <= i < 50:
            econ.simulate(i, True, min_utilisations)
        else:
            econ.simulate(i, False, max_utilisations)
    df = (
        pd.DataFrame(
            [econ.results.fraction_gdp_by_sector(i) for i in range(1, length)],
            index=range(1, length),
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

    # Plot 4
    pd.DataFrame(
        [
            {
                r: econ.results.personal_bankruptcy[i][r].corporate_bankruptcy
                for r in Region
            }
            for i in econ.results.personal_bankruptcy
        ]
    ).plot(figsize=(20, 10))
