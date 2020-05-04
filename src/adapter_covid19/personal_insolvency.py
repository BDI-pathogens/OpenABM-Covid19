import itertools
from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats

from adapter_covid19.constants import START_OF_TIME, DAYS_IN_A_YEAR
from adapter_covid19.datasources import (
    Reader,
    RegionDataSource,
    RegionSectorAgeDataSource,
    RegionDecileSource,
    RegionSectorDecileSource,
)
from adapter_covid19.enums import LabourState, Age
from adapter_covid19.enums import Region, Sector, Decile


@dataclass
class PersonalBankruptcyResults:
    time: int
    delta_balance: Mapping[Sector, Mapping[Decile, float]]
    balance: Mapping[Sector, Mapping[Decile, float]]
    credit_mean: Mapping[Sector, Mapping[Decile, float]]
    credit_std: float
    utilisation: Mapping[Sector, Mapping[LabourState, float]]
    min_expense_cut: Mapping[Sector, Mapping[Decile, float]]
    personal_bankruptcy: float


# FIXME: this shouldn't be a dataclass
@dataclass
class PersonalBankruptcyModel:
    # Threshold of credit score for default
    default_th: float = np.random.rand() * 500

    # Maximum salary when furloughed
    max_earning_furloughed: float = np.random.rand() * 30000

    # Coefficient in credit score regression
    beta: float = np.random.rand() * 10

    # GDP per region per sector per age
    gdp_data: Mapping[Tuple[Region, Sector, Age], float] = field(default=None)

    # Workers per region per sector per age
    workers_data: Mapping[Tuple[Region, Sector, Age], float] = field(default=None)

    # Workers per region
    workers_per_region: Mapping[Region, float] = field(default=None)

    # Workers per region, sector
    workers_per_region_sector: Mapping[Tuple[Region, Sector], float] = field(
        default=None
    )

    # Credit mean by region
    credit_mean: Mapping[Region, float] = field(default=None)

    # Credit std
    credit_std: Mapping[Region, float] = field(default=None)

    # Earnings by region, decile
    earnings: Mapping[Tuple[Region, Sector, Decile], float] = field(default=None)

    # Minimum expenses by region, decile
    expenses: Mapping[Tuple[Region, Decile], float] = field(default=None)

    # Saving by region, decile
    cash_reserve: Mapping[Tuple[Region, Sector, Decile], float] = field(default=None)

    # Earning ratio per labour state
    eta: MutableMapping[LabourState, float] = field(default=None)

    # Sector weightings per region
    _sector_region_weights: Mapping[Region, Mapping[Sector, float]] = field(
        default=None, init=False
    )

    # Results, t by region by PersonalBankruptcyResults
    results: MutableMapping[
        int, MutableMapping[Region, PersonalBankruptcyResults]
    ] = field(default_factory=dict, init=False)

    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def load(self, reader: Reader) -> None:
        if self.gdp_data is None:
            self.gdp_data = RegionSectorAgeDataSource("gdp").load(reader)

        df_gdp = (
            pd.Series(self.gdp_data)
            .to_frame()
            .reset_index()
            .groupby(["level_0", "level_1"])[0]
            .sum()
            .unstack()
        )
        self._sector_region_weights = (df_gdp.T / df_gdp.T.sum(axis=0)).to_dict()

        if self.workers_data is None:
            self.workers_data = RegionSectorAgeDataSource("workers").load(reader)

        self.workers_per_region_sector = {
            (r, s): sum(self.workers_data[r, s, a] for a in Age)
            for r, s in itertools.product(Region, Sector)
        }

        self.workers_per_region = {
            r: sum(self.workers_per_region_sector[r, s] for s in Sector) for r in Region
        }

        if self.credit_mean is None or self.credit_std is None:
            credit_score = RegionDataSource("credit_score").load(reader)
            if self.credit_mean is None:
                self.credit_mean = credit_score["mean"]
            if self.credit_std is None:
                self.credit_std = credit_score["stdev"]

        if self.earnings is None:
            self.earnings = RegionSectorDecileSource("earnings").load(reader)

        if self.expenses is None:
            self.expenses = RegionDecileSource("expenses").load(reader)

        if self.cash_reserve is None:
            self._init_cash_reserve()

        if self.eta is None:
            self._init_eta()

        self._check_data()

    def _check_data(self) -> None:
        for source in [
            self.credit_mean,
            self.credit_std,
        ]:
            regions = set(source.keys())
            if regions != set(Region):
                raise ValueError(f"Inconsistent data: {regions}, {set(Region)}")

    def _init_cash_reserve(self) -> None:
        n_month_cash_reserve = 0.0
        self.cash_reserve = {
            (r, s, d): (self.earnings[(r, s, d)] - self.expenses[(r, d)])
            * n_month_cash_reserve
            / 12.0
            for r, s, d in itertools.product(Region, Sector, Decile)
        }

    def _init_eta(self) -> None:
        self.eta = {
            LabourState.ILL: 1,
            LabourState.WFH: 1,
            LabourState.WORKING: 1,
            LabourState.FURLOUGHED: 0.8,
            LabourState.UNEMPLOYED: 0,
        }

    def simulate(
        self,
        time: int,
        utilisations: Mapping[Tuple[LabourState, Region, Sector, Age], float],
        **kwargs,
    ) -> None:
        # TODO: This is inefficient
        utilisations_h = {
            (r, s): {
                l: sum(
                    utilisations[l, r, s, a] * self.workers_data[r, s, a] for a in Age
                )
                / self.workers_per_region_sector[(r, s)]
                for l in LabourState
            }
            for r, s in itertools.product(Region, Sector)
        }
        self.results[time] = {}
        for r in Region:
            delta_balance_r, balance_r, spot_credit_mean_r, min_expense_cut_r = (
                {},
                {},
                {},
                {},
            )
            for s in Sector:
                if time == START_OF_TIME:
                    delta_balance_rs = {d: 0 for d in Decile}
                    balance_rs = {d: self.cash_reserve[(r, s, d)] for d in Decile}
                else:
                    delta_balance_rs = self._calc_delta_balance(r, s, utilisations_h)
                    balance_rs = {
                        d: self.results[time - 1][r].balance[s][d] + delta_balance_rs[d]
                        for d in Decile
                    }

                spot_credit_mean_rs = self._calc_credit_mean(
                    self.credit_mean[r], balance_rs
                )

                min_expense_cut_rs = self._calc_expense_min_cut(r, delta_balance_rs)

                delta_balance_r[s] = delta_balance_rs
                balance_r[s] = balance_rs
                spot_credit_mean_r[s] = spot_credit_mean_rs
                min_expense_cut_r[s] = min_expense_cut_rs

            personal_bankruptcy_r = self._calc_personal_bankruptcy(
                r, spot_credit_mean_r
            )

            self.results[time][r] = PersonalBankruptcyResults(
                time=time,
                delta_balance=delta_balance_r,
                balance=balance_r,
                credit_mean=spot_credit_mean_r,
                credit_std=self.credit_std[r],
                utilisation={s: utilisations_h[(r, s)] for s in Sector},
                personal_bankruptcy=personal_bankruptcy_r,
                min_expense_cut=min_expense_cut_r,
            )

    def _calc_delta_balance(
        self,
        r: Region,
        s: Sector,
        utilisations_h: Mapping[Tuple[Region, Sector], Mapping[LabourState, float]],
    ) -> Mapping[Decile, float]:
        db = {}
        for d in Decile:
            db_d = 0
            for ls in LabourState:
                spot_earnings = self.eta[ls] * self.earnings[(r, s, d)]
                if ls == LabourState.FURLOUGHED:
                    spot_earnings = min(spot_earnings, self.max_earning_furloughed)

                db_d += (
                    utilisations_h[(r, s)][ls]
                    * (spot_earnings - self.expenses[(r, d)])
                    / DAYS_IN_A_YEAR
                )

            db[d] = db_d
        return db

    def _calc_expense_min_cut(
        self, r: Region, delta_balance: Mapping[Decile, float]
    ) -> Mapping[Decile, float]:
        min_cut = {}
        for d in Decile:
            gap = min(delta_balance[d], 0)
            min_cut[d] = -gap / (self.expenses[(r, d)] / DAYS_IN_A_YEAR)
        return min_cut

    def _calc_credit_mean(
        self, init_credit_mean: float, balance: Mapping[Decile, float],
    ) -> Mapping[Decile, float]:
        return {d: init_credit_mean + self.beta * min(balance[d], 0) for d in Decile}

    def _calc_personal_bankruptcy(
        self, r: Region, spot_credit_mean: Mapping[Sector, Mapping[Decile, float]],
    ) -> float:
        ppb = 0
        decile_weight = 1.0 / len(Decile)
        for s in Sector:
            for d in Decile:
                mixture_weight = (
                    decile_weight
                    * self.workers_per_region_sector[(r, s)]
                    / self.workers_per_region[r]
                )
                ppb += mixture_weight * scipy.stats.norm.cdf(
                    self.default_th, spot_credit_mean[s][d], self.credit_std[r]
                )

        return ppb
