import copy
import itertools
import logging
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
    RegionSectorDecileSource,
)
from adapter_covid19.enums import LabourState, Age
from adapter_covid19.enums import Region, Sector, Decile
from adapter_covid19.data_structures import (
    SimulateState,
    PersonalStateToDeprecate,
    PersonalState,
)

LOGGER = logging.getLogger(__name__)


# FIXME: this shouldn't be a dataclass
@dataclass
class PersonalBankruptcyModel:
    # Threshold of credit score for default
    default_th: float = np.random.rand() * 500

    # Maximum salary when furloughed
    max_earning_furloughed: float = np.random.rand() * 30000

    # Coefficient in credit score regression on spot balance delta
    alpha: float = np.random.rand() * 10

    # Coefficient in credit score regression on spot balance
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

    # Detailed expenses by region, employed sector, decile, spending sector
    detailed_expenses: Mapping[Tuple[Region, Sector, Decile, Sector], float] = field(
        default=None
    )

    # Expenses by region, employed sector, decile
    expenses: MutableMapping[Tuple[Region, Sector, Decile], float] = field(
        default_factory=dict, init=False
    )

    # Expenses by expense sector, decile
    expenses_by_expense_sector: MutableMapping[Sector, float] = field(
        default_factory=dict, init=False
    )

    # Saving by region, decile
    cash_reserve: Mapping[Tuple[Region, Sector, Decile], float] = field(default=None)

    # Earning ratio per labour state
    eta: MutableMapping[LabourState, float] = field(default=None)

    # Sector weightings per region
    _sector_region_weights: Mapping[Region, Mapping[Sector, float]] = field(
        default=None, init=False
    )

    # Results, t by region by PersonalBankruptcyResults
    results: MutableMapping[int, PersonalState] = field(
        default_factory=dict, init=False
    )

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

        if self.detailed_expenses is None:
            self._init_detailed_expenses(reader)

        self.expenses = {
            (region, employed_sector, decile): sum(
                self.detailed_expenses[region, employed_sector, decile, expense_sector]
                for expense_sector in Sector
            )
            for region, employed_sector, decile in itertools.product(
                Region, Sector, Decile
            )
        }

        self.expenses_by_expense_sector = {
            expense_sector: sum(
                self.detailed_expenses[region, employed_sector, decile, expense_sector]
                for region, employed_sector, decile in itertools.product(
                    Region, Sector, Decile
                )
            )
            for expense_sector in Sector
        }

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

    def _init_detailed_expenses(self, reader: Reader) -> None:
        expenses_by_region_expense_sector_decile = RegionSectorDecileSource(
            "expenses"
        ).load(reader)

        self.detailed_expenses = {}

        for region, decile in itertools.product(Region, Decile):
            earnings_rd = {
                employed_sector: self.earnings[(region, employed_sector, decile)]
                for employed_sector in Sector
            }
            earnings_rd_avg = np.mean(list(earnings_rd.values()))
            for employed_sector, expense_sector in itertools.product(Sector, Sector):
                self.detailed_expenses[
                    (region, employed_sector, decile, expense_sector)
                ] = (
                        expenses_by_region_expense_sector_decile[
                            (region, expense_sector, decile)
                        ]
                        * earnings_rd[employed_sector]
                        / earnings_rd_avg
                )

    def _init_cash_reserve(self) -> None:
        n_month_cash_reserve = 0.0
        self.cash_reserve = {
            (r, s, d): (self.earnings[(r, s, d)] - self.expenses[(r, s, d)])
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

    def simulate(self, state: SimulateState) -> None:
        # TODO: This is inefficient
        utilisations_h = {
            (r, s): {
                l: sum(
                    state.utilisations[l, r, s, a] * self.workers_data[r, s, a]
                    for a in Age
                )
                   / self.workers_per_region_sector[(r, s)]
                for l in LabourState
            }
            for r, s in itertools.product(Region, Sector)
        }

        personal_state = PersonalState(
            time=state.time,
            delta_balance={},
            balance={},
            credit_mean={},
            credit_std={},
            utilisation=utilisations_h,
            personal_bankruptcy={},
            demand_reduction={},
        )

        for r in Region:
            spot_credit_mean_r = {}
            for s, d in itertools.product(Sector, Decile):
                if state.time == START_OF_TIME:
                    delta_balance_rsd = 0
                    balance_rsd = self.cash_reserve[(r, s, d)]
                else:
                    delta_balance_rsd = self._calc_delta_balance(
                        r, s, d, utilisations_h
                    )

                    # if kwargs.get('update_balance', True):
                    #     delta_balance_rs = self._calc_delta_balance(r, s, d, utilisations_h)
                    # else:
                    #     delta_balance_rs = {d: 0 for d in Decile}

                    balance_rsd = (
                            state.previous.personal_state.balance[r, s, d]
                            + delta_balance_rsd
                    )

                spot_credit_mean_rsd = self._calc_credit_mean(
                    r, delta_balance_rsd, balance_rsd
                )

                spot_credit_mean_r[(s, d)] = spot_credit_mean_rsd

                personal_state.delta_balance[(r, s, d)] = delta_balance_rsd
                personal_state.balance[(r, s, d)] = balance_rsd
                personal_state.credit_mean[(r, s, d)] = spot_credit_mean_rsd
                personal_state.credit_std[(r, s, d)] = self.credit_std[r]
                personal_state.utilisation = utilisations_h

            personal_state.personal_bankruptcy[r] = self._calc_personal_bankruptcy(
                r, spot_credit_mean_r
            )

        demand_reduction = self._calc_demand_reduction(personal_state.delta_balance)
        personal_state.demand_reduction = demand_reduction

        state.personal_state = copy.deepcopy(personal_state)
        self.results[state.time] = copy.deepcopy(personal_state)

    def _calc_delta_balance(
            self,
            r: Region,
            s: Sector,
            d: Decile,
            utilisations_h: Mapping[Tuple[Region, Sector], Mapping[LabourState, float]],
    ) -> float:
        delta_balance = 0
        for ls in LabourState:
            spot_earnings = self.eta[ls] * self.earnings[(r, s, d)]
            if ls == LabourState.FURLOUGHED:
                spot_earnings = min(spot_earnings, self.max_earning_furloughed)

            delta_balance += (
                    utilisations_h[(r, s)][ls]
                    * (spot_earnings - self.expenses[(r, s, d)])
                    / DAYS_IN_A_YEAR
            )
        return delta_balance

    def _calc_credit_mean(
            self, r: Region, delta_balance: float, balance: float,
    ) -> float:
        return (
                self.credit_mean[r]
                + self.alpha * delta_balance
                + self.beta * min(balance, 0)
        )

    def _calc_personal_bankruptcy(
            self, r: Region, spot_credit_mean: Mapping[Tuple[Sector, Decile], float],
    ) -> float:
        ppb = 0
        decile_weight = 1.0 / len(Decile)
        for s, d in itertools.product(Sector, Decile):
            mixture_weight = (
                    decile_weight
                    * self.workers_per_region_sector[(r, s)]
                    / self.workers_per_region[r]
            )

            if spot_credit_mean[(s, d)] > 0:
                cdf_per_mixture_sd = scipy.stats.norm.cdf(
                    self.default_th, spot_credit_mean[(s, d)], self.credit_std[r]
                )
            else:
                cdf_per_mixture_sd = 1.0

            ppb += mixture_weight * cdf_per_mixture_sd

        return ppb

    def _calc_demand_reduction(
            self, delta_balance: Mapping[Tuple[Region, Sector, Decile], float],
    ) -> Mapping[Sector, float]:
        demand_reduction = {expense_sector: 0.0 for expense_sector in Sector}
        for region in Region:
            for employed_sector in Sector:
                for decile in Decile:
                    min_cut_rsd = -min(
                        delta_balance[(region, employed_sector, decile)], 0
                    ) / (
                                          self.expenses[(region, employed_sector, decile)]
                                          / DAYS_IN_A_YEAR
                                  )

                    for expense_sector in Sector:
                        demand_reduction[expense_sector] += (
                                min_cut_rsd
                                * self.detailed_expenses[
                                    (region, employed_sector, decile, expense_sector)
                                ]
                                / self.expenses_by_expense_sector[expense_sector]
                        )

        return demand_reduction
