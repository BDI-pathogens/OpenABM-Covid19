import copy
import itertools
import logging
from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Tuple, Dict

import numpy as np
import pandas as pd
import scipy.stats

from adapter_covid19.constants import START_OF_TIME, DAYS_IN_A_YEAR
from adapter_covid19.data_structures import (
    SimulateState,
    PersonalState,
    Utilisations,
)
from adapter_covid19.datasources import (
    Reader,
    RegionDataSource,
    RegionSectorAgeDataSource,
    RegionSectorDecileSource,
)
from adapter_covid19.enums import Age, WorkerState
from adapter_covid19.enums import Region, Sector, Decile

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

    # Credit mean by region
    credit_mean: Mapping[Region, float] = field(default=None)

    # Credit std
    credit_std: Mapping[Region, float] = field(default=None)

    # Daily earnings by region, sector, decile
    earnings: Mapping[Tuple[Region, Sector, Decile], float] = field(default=None)

    # Daily earnings by region, sector, decile, worker state
    _earnings_by_worker_state: Mapping[
        Tuple[Region, Sector, Decile, WorkerState], float
    ] = field(default=None, init=False)

    # Daily detailed expenses by region, employed sector, decile, spending sector
    detailed_expenses: Mapping[Tuple[Region, Sector, Decile, Sector], float] = field(
        default=None
    )

    # Daily expenses by region, employed sector, decile
    expenses: MutableMapping[Tuple[Region, Sector, Decile], float] = field(
        default_factory=dict, init=False
    )

    # Daily expenses by expense sector, decile
    expenses_by_expense_sector: MutableMapping[Sector, float] = field(
        default_factory=dict, init=False
    )

    # Daily detailed minimum expenses by region, employed sector, decile, spending sector
    detailed_min_expenses: Mapping[
        Tuple[Region, Sector, Decile, Sector], float
    ] = field(default=None)

    # Saving by region, decile
    cash_reserve: Mapping[Tuple[Region, Sector, Decile], float] = field(default=None)

    # Earning ratio per labour state
    eta: MutableMapping[WorkerState, float] = field(default=None)

    # Mixture weights per  region, sector
    _mixture_weights: Mapping[Tuple[Region, Sector, Decile], float] = field(
        default=None, init=False
    )

    # CFD cache to make computation faster
    _cdf_cache: Dict[float, float] = field(default_factory=dict, init=False)

    # Sector weightings per region
    _sector_region_weights: Mapping[Region, Mapping[Sector, float]] = field(
        default=None, init=False
    )

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

        _workers_per_region_sector = {
            (r, s): sum(self.workers_data[r, s, a] for a in Age)
            for r, s in itertools.product(Region, Sector)
        }

        _workers_per_region = {
            r: sum(_workers_per_region_sector[r, s] for s in Sector) for r in Region
        }

        _decile_weight = 1.0 / len(Decile)
        self._mixture_weights = {
            (r, s, d): _decile_weight
            * _workers_per_region_sector[(r, s)]
            / _workers_per_region[r]
            for r, s, d in itertools.product(Region, Sector, Decile)
        }

        if self.credit_mean is None or self.credit_std is None:
            credit_score = RegionDataSource("credit_score").load(reader)
            if self.credit_mean is None:
                self.credit_mean = credit_score["mean"]
            if self.credit_std is None:
                self.credit_std = credit_score["stdev"]

        if self.earnings is None:
            self._init_earnings(reader)

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

        if self.detailed_min_expenses is None:
            self._init_detailed_min_expenses(reader)

        if self.cash_reserve is None:
            self._init_cash_reserve()

        if self.eta is None:
            self._init_eta()

        self._earnings_by_worker_state = {}
        for r, s, d, ws in itertools.product(Region, Sector, Decile, WorkerState):
            _earnings = self.eta[ws] * self.earnings[(r, s, d)]
            if ws in {
                WorkerState.HEALTHY_FURLOUGHED,
                WorkerState.ILL_FURLOUGHED,
            }:
                _earnings = min(_earnings, self.max_earning_furloughed)
            self._earnings_by_worker_state[(r, s, d, ws)] = _earnings

        self._check_data()

    def _check_data(self) -> None:
        for source in [
            self.credit_mean,
            self.credit_std,
        ]:
            regions = set(source.keys())
            if regions != set(Region):
                raise ValueError(f"Inconsistent data: {regions}, {set(Region)}")

    def _init_earnings(self, reader: Reader) -> None:
        annual_earnings = RegionSectorDecileSource("earnings").load(reader)

        self.earnings = {
            (r, s, d): annual_earnings[(r, s, d)] / DAYS_IN_A_YEAR
            for r, s, d in itertools.product(Region, Sector, Decile)
        }

    def _init_detailed_expenses(self, reader: Reader) -> None:
        expenses_by_region_expense_sector_decile = RegionSectorDecileSource(
            "expenses_full"
        ).load(reader)

        self.detailed_expenses = self._extend_deciles_to_expenses(
            expenses_by_region_expense_sector_decile
        )

    def _init_detailed_min_expenses(self, reader: Reader) -> None:
        min_expenses_by_region_expense_sector_decile = RegionSectorDecileSource(
            "min_expenses_full"
        ).load(reader)

        self.detailed_min_expenses = self._extend_deciles_to_expenses(
            min_expenses_by_region_expense_sector_decile
        )

    def _extend_deciles_to_expenses(
        self,
        expenses_by_region_expense_sector_decile: Mapping[
            Tuple[Region, Sector, Decile], float
        ],
    ) -> Mapping[Tuple[Region, Sector, Decile, Sector], float]:
        extended_expenses = {}

        for region, decile in itertools.product(Region, Decile):
            earnings_rd = {
                employed_sector: self.earnings[(region, employed_sector, decile)]
                for employed_sector in Sector
            }
            earnings_rd_avg = np.mean(list(earnings_rd.values()))
            for employed_sector, expense_sector in itertools.product(Sector, Sector):
                extended_expenses[(region, employed_sector, decile, expense_sector)] = (
                    expenses_by_region_expense_sector_decile[
                        (region, expense_sector, decile)
                    ]
                    * earnings_rd[employed_sector]
                    / earnings_rd_avg
                    / DAYS_IN_A_YEAR
                )
        return extended_expenses

    def _init_cash_reserve(self) -> None:
        n_month_cash_reserve = 0.0
        self.cash_reserve = {
            (r, s, d): (self.earnings[(r, s, d)] - self.expenses[(r, s, d)])
            * n_month_cash_reserve
            * 30
            for r, s, d in itertools.product(Region, Sector, Decile)
        }

    def _init_eta(self) -> None:
        self.eta = {
            WorkerState.HEALTHY_WFO: 1,
            WorkerState.HEALTHY_WFH: 1,
            WorkerState.HEALTHY_FURLOUGHED: 0.8,
            WorkerState.HEALTHY_UNEMPLOYED: 0,
            WorkerState.ILL_WFO: 1,
            WorkerState.ILL_WFH: 1,
            WorkerState.ILL_FURLOUGHED: 0.8,
            WorkerState.ILL_UNEMPLOYED: 0,
            WorkerState.DEAD: 0,
        }

    def simulate(self, state: SimulateState) -> None:
        personal_state = PersonalState(
            time=state.time,
            spot_earning={},
            spot_expense={},
            spot_expense_by_sector={},
            delta_balance={},
            balance={},
            credit_mean={},
            credit_std={},
            personal_bankruptcy={},
            demand_reduction={},
        )
        for region in Region:
            spot_credit_mean_r = {}
            for employed_sector, decile in itertools.product(Sector, Decile):
                if state.time == START_OF_TIME:
                    starting_balance_rsd = self.cash_reserve[
                        (region, employed_sector, decile)
                    ]
                else:
                    starting_balance_rsd = state.previous.personal_state.balance[
                        (region, employed_sector, decile)
                    ]

                spot_earning_rsd = self._calc_spot_earning(
                    region, employed_sector, decile, state.utilisations
                )

                spot_expense_by_sector_rsd = self._calc_spot_expense_by_sector(
                    region, employed_sector, decile, spot_earning_rsd
                )

                spot_expense_rsd = sum(spot_expense_by_sector_rsd.values())
                delta_balance_rsd = spot_earning_rsd - spot_expense_rsd
                balance_rsd = starting_balance_rsd + delta_balance_rsd

                spot_credit_mean_rsd = self._calc_credit_mean(
                    region, delta_balance_rsd, balance_rsd
                )

                spot_credit_mean_r[(employed_sector, decile)] = spot_credit_mean_rsd

                personal_state.spot_earning[
                    (region, employed_sector, decile)
                ] = spot_earning_rsd

                for expense_sector in Sector:
                    personal_state.spot_expense_by_sector[
                        (region, employed_sector, decile, expense_sector)
                    ] = spot_expense_by_sector_rsd[expense_sector]

                personal_state.spot_expense[
                    (region, employed_sector, decile)
                ] = spot_expense_rsd
                personal_state.delta_balance[
                    (region, employed_sector, decile)
                ] = delta_balance_rsd
                personal_state.balance[(region, employed_sector, decile)] = balance_rsd
                personal_state.credit_mean[
                    (region, employed_sector, decile)
                ] = spot_credit_mean_rsd
                personal_state.credit_std[
                    (region, employed_sector, decile)
                ] = self.credit_std[region]

            personal_state.personal_bankruptcy[region] = self._calc_personal_bankruptcy(
                region, spot_credit_mean_r
            )

        demand_reduction = self._calc_demand_reduction(
            personal_state.spot_expense_by_sector
        )
        personal_state.demand_reduction = demand_reduction

        state.personal_state = copy.deepcopy(personal_state)

        # self.results[state.time] = copy.deepcopy(personal_state)

    def _calc_spot_earning(
        self, r: Region, s: Sector, d: Decile, utilisations: Utilisations
    ) -> float:
        spot_earning = 0
        for worker_state in WorkerState:
            spot_earning += (
                utilisations[r, s][worker_state]
                * self._earnings_by_worker_state[(r, s, d, worker_state)]
            )
        return spot_earning

    def _calc_spot_expense_by_sector(
        self,
        region: Region,
        employed_sector: Sector,
        decile: Decile,
        spot_earning: float,
    ) -> Mapping[Sector, float]:
        spot_earning_ratio = min(
            spot_earning / self.earnings[(region, employed_sector, decile)], 1.0
        )
        return {
            expense_sector: max(
                self.detailed_expenses[
                    (region, employed_sector, decile, expense_sector)
                ]
                * spot_earning_ratio,
                self.detailed_min_expenses[
                    (region, employed_sector, decile, expense_sector)
                ],
            )
            for expense_sector in Sector
        }

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
        for s, d in itertools.product(Sector, Decile):
            ppb += self._mixture_weights[(r, s, d)] * self._get_cdf(
                self.default_th, spot_credit_mean[(s, d)], self.credit_std[r]
            )

        return ppb

    def _get_cdf(self, v: float, mu: float, std: float) -> float:
        z_score = max(min(round_to_half_int(v - mu / std), 3), -3)
        try:
            return self._cdf_cache[z_score]
        except KeyError:
            cdf = scipy.stats.norm.cdf(v, mu, std)
            self._cdf_cache[z_score] = cdf
            return cdf

    def _calc_demand_reduction(
        self,
        spot_expense_by_sector: Mapping[Tuple[Region, Sector, Decile, Sector], float],
    ) -> Mapping[Sector, float]:
        expense_by_expense_sector = {
            expense_sector: sum(
                spot_expense_by_sector[region, exmployed_sector, decile, expense_sector]
                for region, exmployed_sector, decile in itertools.product(
                    Region, Sector, Decile
                )
            )
            for expense_sector in Sector
        }

        demand_reduction = {
            expense_sector: (
                1
                - expense_by_expense_sector[expense_sector]
                / self.expenses_by_expense_sector[expense_sector]
                if self.expenses_by_expense_sector[expense_sector] > 0
                else 0.0
            )
            for expense_sector in Sector
        }

        return demand_reduction


def round_to_half_int(number: float):
    """Round a number to the closest half integer."""

    return round(number * 2) / 2
