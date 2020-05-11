from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats

from adapter_covid19.constants import START_OF_TIME, DAYS_IN_A_YEAR
from adapter_covid19.datasources import (
    Reader,
    RegionDataSource,
    RegionSectorAgeDataSource,
)
from adapter_covid19.enums import LabourState, Age
from adapter_covid19.enums import Region, Sector


@dataclass
class PersonalBankruptcyResults:
    time: int
    earning: Mapping[LabourState, float]
    balance: Mapping[LabourState, float]
    credit_mean: Mapping[LabourState, float]
    credit_std: float
    utilisations: Mapping[LabourState, float]
    personal_bankruptcy: float
    corporate_bankruptcy: float


@dataclass
class PersonalBankruptcyModel:
    # Individual saving
    saving: float = np.random.rand() * 1000

    # Threshold of credit score for default
    default_th: float = np.random.rand() * 500

    # Coefficient in credit score regression
    beta: float = np.random.rand() * 10

    # Percentage of earnings in non-essential expense
    gamma: float = np.random.rand()

    # Ratios in simulating utilization factors
    utilization_ratio_ill: float = np.random.rand()
    utilization_ratio_furloughed_lockdown: float = None
    utilization_ratio_wfh_lockdown: float = None
    utilization_ratio_working_lockdown: float = None
    utilization_ratio_furloughed_no_lockdown: float = None
    utilization_ratio_wfh_no_lockdown: float = None
    utilization_ratio_working_no_lockdown: float = None

    # GDP per region per sector per age
    gdp_data: Mapping[Tuple[Region, Sector, Age], float] = field(
        default=None, init=False
    )

    # Sector weightings per region
    sector_region_weights: Mapping[Region, Mapping[Sector, float]] = field(
        default=None, init=False
    )

    # Credit mean by region
    init_credit_mean: Mapping[Region, float] = field(default=None, init=False)

    # Credit std by region
    credit_std: Mapping[Region, float] = field(default=None, init=False)

    # Earnings by region
    init_earning: Mapping[Region, float] = field(default=None, init=False)

    # Earning ratio per labour state
    eta: MutableMapping[LabourState, float] = field(default_factory=dict, init=False)

    # Minimum expenses by region
    min_expense: Mapping[Region, float] = field(default=None, init=False)

    # Saving by region
    init_saving: Mapping[Region, float] = field(default=None, init=False)

    # Cache to compute moving average of utilization
    utilization_cache: List[Mapping[Region, Mapping[LabourState, float]]] = field(
        default_factory=list, init=False
    )

    # Window size of moving average of the utilization factor
    utilization_ma_win: int = 20

    # Results, t by region by PersonalBankruptcyResults
    results: MutableMapping[
        int, MutableMapping[Region, PersonalBankruptcyResults]
    ] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.eta[LabourState.ill] = 1
        self.eta[LabourState.wfh] = 1
        self.eta[LabourState.working] = 1
        self.eta[LabourState.furloughed] = 0.8
        self.eta[LabourState.unemployed] = 0

        if (
            self.utilization_ratio_furloughed_lockdown is None
            or self.utilization_ratio_wfh_lockdown is None
            or self.utilization_ratio_working_lockdown is None
        ):
            (
                self.utilization_ratio_furloughed_lockdown,
                self.utilization_ratio_wfh_lockdown,
                self.utilization_ratio_working_lockdown,
            ) = np.random.dirichlet(([1, 1, 1]))

        np.testing.assert_almost_equal(
            self.utilization_ratio_furloughed_lockdown
            + self.utilization_ratio_wfh_lockdown
            + self.utilization_ratio_working_lockdown,
            1.0,
            decimal=4,
        )

        if (
            self.utilization_ratio_furloughed_no_lockdown is None
            or self.utilization_ratio_wfh_no_lockdown is None
            or self.utilization_ratio_working_no_lockdown is None
        ):
            (
                self.utilization_ratio_furloughed_no_lockdown,
                self.utilization_ratio_wfh_no_lockdown,
                self.utilization_ratio_working_no_lockdown,
            ) = np.random.dirichlet(([1, 1, 1]))

        np.testing.assert_almost_equal(
            self.utilization_ratio_furloughed_no_lockdown
            + self.utilization_ratio_wfh_no_lockdown
            + self.utilization_ratio_working_no_lockdown,
            1.0,
            decimal=4,
        )

    def _check_data(self) -> None:
        for source in [
            self.init_credit_mean,
            self.credit_std,
            self.init_earning,
            self.min_expense,
        ]:
            regions = set(source.keys())
            if regions != set(Region):
                raise ValueError(f"Inconsistent data: {regions}, {set(Region)}")

    def load(self, reader: Reader) -> None:
        self.gdp_data = RegionSectorAgeDataSource("gdp").load(reader)
        df_gdp = (
            pd.Series(self.gdp_data)
            .to_frame()
            .reset_index()
            .groupby(["level_0", "level_1"])[0]
            .sum()
            .unstack()
        )
        self.sector_region_weights = (df_gdp.T / df_gdp.T.sum(axis=0)).to_dict()

        credit_score = RegionDataSource("credit_score").load(reader)
        self.init_credit_mean = credit_score["mean"]
        self.credit_std = credit_score["stdev"]
        self.init_earning = RegionDataSource("earnings").load(reader)
        self.min_expense = RegionDataSource("expenses").load(reader)
        self._check_data()
        self.init_saving = self._get_init_saving()

    def simulate(
        self, time: int, lockdown: bool, corporate_bankruptcy: Mapping[Sector, float],
    ) -> None:
        corporate_bankruptcy = {
            r: sum(
                v * self.sector_region_weights[r][s]
                for s, v in corporate_bankruptcy.items()
            )
            for r in Region
        }

        utilisations = self.get_utilization(lockdown, corporate_bankruptcy)
        self.utilization_cache.append(utilisations)
        if len(self.utilization_cache) > self.utilization_ma_win:
            self.utilization_cache.pop(0)

        if time == START_OF_TIME:
            self.results[time] = {}
            for r in Region:
                earning = self._calc_earning(self.init_earning[r])

                balance = {ls: self.init_saving[r] for ls in LabourState}

                credit_mean = {ls: self.init_credit_mean[r] for ls in LabourState}

                self.utilization_cache.append(utilisations)

                personal_bankruptcy = self._calc_personal_bankruptcy(
                    r, credit_mean, self.credit_std[r]
                )

                self.results[time][r] = PersonalBankruptcyResults(
                    time=time,
                    earning=earning,
                    balance=balance,
                    credit_mean=credit_mean,
                    credit_std=self.credit_std[r],
                    utilisations=utilisations[r],
                    personal_bankruptcy=personal_bankruptcy,
                    corporate_bankruptcy=corporate_bankruptcy[r],
                )
            return

        self.results[time] = {}
        for r in Region:
            earning = self._calc_earning(self.init_earning[r])

            balance = self._calc_balance(
                self.results[time - 1][r].balance, earning, self.min_expense[r]
            )

            credit_mean = self._calc_credit_mean(self.init_credit_mean[r], balance)

            personal_bankruptcy = self._calc_personal_bankruptcy(
                r, credit_mean, self.credit_std[r]
            )

            self.results[time][r] = PersonalBankruptcyResults(
                time=time,
                earning=earning,
                balance=balance,
                credit_mean=credit_mean,
                credit_std=self.credit_std[r],
                utilisations=utilisations[r],
                personal_bankruptcy=personal_bankruptcy,
                corporate_bankruptcy=corporate_bankruptcy[r],
            )

    def get_utilization(
        self, lockdown: bool, corporate_bankruptcy: Mapping[Region, float],
    ) -> Mapping[Region, Mapping[LabourState, float]]:
        utilization = {}

        for r in Region:
            utilization_r = {}
            utilization_r_sum = 0
            # We first lock lambda_unemployed
            utilization_r[LabourState.unemployed] = corporate_bankruptcy[r]
            utilization_r_sum += utilization_r[LabourState.unemployed]

            # Next we check lambda_ill
            utilization_r[LabourState.ill] = min(
                self.utilization_ratio_ill, 1 - utilization_r_sum
            )
            utilization_r_sum += utilization_r[LabourState.ill]

            # Next we check furloughed, wfh and working
            if lockdown:
                utilization_r[
                    LabourState.furloughed
                ] = self.utilization_ratio_furloughed_lockdown * (1 - utilization_r_sum)
                utilization_r[LabourState.wfh] = self.utilization_ratio_wfh_lockdown * (
                    1 - utilization_r_sum
                )
                utilization_r[
                    LabourState.working
                ] = self.utilization_ratio_working_lockdown * (1 - utilization_r_sum)
            else:
                utilization_r[
                    LabourState.furloughed
                ] = self.utilization_ratio_furloughed_no_lockdown
                utilization_r[
                    LabourState.wfh
                ] = self.utilization_ratio_wfh_no_lockdown * (1 - utilization_r_sum)
                utilization_r[
                    LabourState.working
                ] = self.utilization_ratio_working_no_lockdown * (1 - utilization_r_sum)

            utilization[r] = utilization_r

        return utilization

    def _calc_personal_bankruptcy(
        self,
        region: Region,
        credit_mean: Mapping[LabourState, float],
        credit_std: float,
    ) -> float:
        dict_util = (
            pd.DataFrame([util_t[region] for util_t in self.utilization_cache])
            .mean()
            .to_dict()
        )

        ppb = 0
        for ls in LabourState:
            ppb += dict_util[ls] * scipy.stats.norm.cdf(
                self.default_th, credit_mean[ls], credit_std
            )

        return ppb

    def _calc_earning(self, init_earning: float) -> Mapping[LabourState, float]:
        return {ls: init_earning * self.eta[ls] for ls in LabourState}

    def _calc_balance(
        self,
        prev_balance: Mapping[LabourState, float],
        earning: Mapping[LabourState, float],
        min_expense: float,
    ) -> Mapping[LabourState, float]:
        return {
            ls: prev_balance[ls]
            + ((1 - self.gamma) * earning[ls] - min_expense) / DAYS_IN_A_YEAR
            for ls in LabourState
        }

    def _calc_credit_mean(
        self, init_credit_mean: float, balance: Mapping[LabourState, float],
    ) -> Mapping[LabourState, float]:
        return {
            ls: init_credit_mean + self.beta * min(balance[ls], 0) for ls in LabourState
        }

    def _get_init_saving(self) -> Dict[Region, float]:
        # TODO: get actual saving figures per region
        return {r: self.saving for r in Region}
