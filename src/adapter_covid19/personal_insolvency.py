import itertools
import logging
from typing import Mapping, MutableMapping, Tuple

import numpy as np
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


class PersonalBankruptcyModel:
    def __init__(
        self,
        default_th: float = np.random.rand() * 500,
        max_earning_furloughed: float = np.random.rand() * 30000,
        alpha: float = np.random.rand() * 10,
        beta: float = np.random.rand() * 10,
        workers_data: Mapping[Tuple[Region, Sector, Age], float] = None,
        credit_mean: Mapping[Region, float] = None,
        credit_std: Mapping[Region, float] = None,
        earnings: Mapping[Tuple[Region, Sector, Decile], float] = None,
        full_expenses: Mapping[Tuple[Region, Sector, Decile, Sector], float] = None,
        full_min_expenses: Mapping[Tuple[Region, Sector, Decile, Sector], float] = None,
        cash_reserve: Mapping[Tuple[Region, Sector, Decile], float] = None,
        eta: MutableMapping[WorkerState, float] = None,
    ):
        """
        Personal bankrupcy model
        :param default_th: Threshold of individual's credit score to be considered as default
        :type: default_th: float
        :param max_earning_furloughed: Maximum salary an employee can receive when furloughed
        :type: max_earning_furloughed: float
        :param alpha: Coefficient of the change of spot balance in credit score regression model
        :type: alpha: float
        :param beta: Coefficient of the spot balance in credit score regression on
        :type: beta:float
        :param workers_data: Workers per region per sector per age
        :type: workers_data: Mapping[Tuple[Region, Sector, Age], float]
        :param credit_mean: Mean of credit score by region
        :type: credit_mean: Mapping[Region, float]
        :param credit_std: Standard deviation of credit std by region
        :type: credit_std: Mapping[Region, float]
        :param earnings: Daily earnings by region, sector, decile
        :type: earnings: Mapping[Tuple[Region, Sector, Decile], float]
        :param full_expenses: Daily expenses by region, employed sector, decile, spending sector. This is the sum of
        three components: 1. basic living costs, such as food and accomodation; 2. civil payments such as tax;
        3. discretionary spending such as holiday
        :type: full_expenses: Mapping[Tuple[Region, Sector, Decile, Sector], float]
        :param full_min_expenses: Similar to full_expenses but only includes basic living costs and civil payments
        :type: full_min_expenses:  Tuple[Region, Sector, Decile, Sector]
        :param cash_reserve: Saving by region, employed sector, decile
        :type: cash_reserve: Mapping[Tuple[Region, Sector, Decile], float]
        :param eta: Earning ratio per labour state, with respect to pre-crisis level
        :type: eta: MutableMapping[WorkerState, float]
        """
        self.default_th = default_th
        self.max_earning_furloughed = max_earning_furloughed
        self.alpha = alpha
        self.beta = beta
        self.workers_data = workers_data
        self.credit_mean = credit_mean
        self.credit_std = credit_std
        self.earnings = earnings
        self.detailed_expenses = full_expenses
        self.detailed_min_expenses = full_min_expenses
        self.cash_reserve = cash_reserve
        self.eta = eta

        # Initialize various cache to improve performance

        # Daily earnings by region, sector, decile, worker state
        self._cache_earnings_by_worker_state = {}

        # Daily expenses by region, employed sector, decile
        self._cache_expenses_by_red = {}

        # Daily expenses by expense sector, decile
        self._cache_expenses_by_expense_sector = {}

        # Mixture weights per region, employed sector, decile
        self._cache_mixture_weights = {}

        # Cache to store CFD values
        self._cache_cdf = {}

    def load(self, reader: Reader) -> None:
        """
        Load data for the ones which are not initialized during the class construction. This method is suggested
        to be called at development only
        """
        if self.workers_data is None:
            self.workers_data = RegionSectorAgeDataSource("workers").load(reader)

        self._update_cache_mixture_weights()

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

        self._update_cache_expenses()

        if self.detailed_min_expenses is None:
            self._init_detailed_min_expenses(reader)

        if self.cash_reserve is None:
            self._init_cash_reserve()

        if self.eta is None:
            self._init_eta()

        self._update_cache_earnings_by_worker_state()

        self._check_data()

    def _update_cache_mixture_weights(self) -> None:
        _workers_per_region_sector = {
            (r, s): sum(self.workers_data[r, s, a] for a in Age)
            for r, s in itertools.product(Region, Sector)
        }

        _workers_per_region = {
            r: sum(_workers_per_region_sector[r, s] for s in Sector) for r in Region
        }

        _decile_weight = 1.0 / len(Decile)
        self._cache_mixture_weights = {
            (r, s, d): _decile_weight
            * _workers_per_region_sector[(r, s)]
            / _workers_per_region[r]
            for r, s, d in itertools.product(Region, Sector, Decile)
        }

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

    def _update_cache_expenses(self) -> None:
        self._cache_expenses_by_red = {
            (region, employed_sector, decile): sum(
                self.detailed_expenses[region, employed_sector, decile, expense_sector]
                for expense_sector in Sector
            )
            for region, employed_sector, decile in itertools.product(
                Region, Sector, Decile
            )
        }

        self._cache_expenses_by_expense_sector = {
            expense_sector: sum(
                self.detailed_expenses[region, employed_sector, decile, expense_sector]
                for region, employed_sector, decile in itertools.product(
                    Region, Sector, Decile
                )
            )
            for expense_sector in Sector
        }

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
            (r, s, d): (
                self.earnings[(r, s, d)] - self._cache_expenses_by_red[(r, s, d)]
            )
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

    def _update_cache_earnings_by_worker_state(self) -> None:
        for r, s, d, ws in itertools.product(Region, Sector, Decile, WorkerState):
            _earnings = self.eta[ws] * self.earnings[(r, s, d)]
            if ws in {
                WorkerState.HEALTHY_FURLOUGHED,
                WorkerState.ILL_FURLOUGHED,
            }:
                _earnings = min(_earnings, self.max_earning_furloughed)
            self._cache_earnings_by_worker_state[(r, s, d, ws)] = _earnings

    def _check_data(self) -> None:
        for source in [
            self.credit_mean,
            self.credit_std,
        ]:
            regions = set(source.keys())
            if regions != set(Region):
                raise ValueError(f"Inconsistent data: {regions}, {set(Region)}")

    def simulate(self, state: SimulateState) -> None:
        """
        Simulate one iteration of the personal bankrupcy model given a simulate state
        :param state: SimulateState
        :return: None
        """
        # Initialize a container of results. It will be attached to the input SimulateState object once this round of
        # simulation is completed
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

        # Load fear factor from SimulateState
        fear_factor = state.get_fear_factor()

        for region in Region:
            # We project personal insolvency ratio to each region. First creat container for mean of credit scores of ]
            # each region
            spot_credit_mean_r = {}
            for employed_sector, decile in itertools.product(Sector, Decile):
                if state.time == START_OF_TIME:
                    # If the current time is the exact start of the simulation, initialize starting balance to be the
                    # corresponding cache reserve
                    starting_balance_rsd = self.cash_reserve[
                        (region, employed_sector, decile)
                    ]
                else:
                    # I f the current time is not the exact start of the simulation, use the balance from the previous
                    # round as the starting balance of this iteration
                    starting_balance_rsd = state.previous.personal_state.balance[
                        (region, employed_sector, decile)
                    ]

                # Compute spot earning, which is how much people can earn during this iteration, with respect to the
                # ratios of each worker state
                spot_earning_rsd = self._calc_spot_earning(
                    region, employed_sector, decile, state.utilisations
                )

                # Compute spot expense, which is how much people are willing to spend during this iteration, after
                # taking into account salary reduction, fear factor and minimum spending
                spot_expense_by_sector_rsd = self._calc_spot_expense_by_sector(
                    region,
                    employed_sector,
                    decile,
                    spot_earning_rsd,
                    fear_factor,
                    state.utilisations,
                )

                # Aggregate spot expenses of each expense sector
                spot_expense_rsd = sum(spot_expense_by_sector_rsd.values())

                # Calculate how much individual operational surplus is expected during this iteration: income - spending
                delta_balance_rsd = spot_earning_rsd - spot_expense_rsd

                # Accumulate balance
                balance_rsd = starting_balance_rsd + delta_balance_rsd

                # Compute the latest mean of credit score
                spot_credit_mean_rsd = self._calc_credit_mean(
                    region, delta_balance_rsd, balance_rsd
                )

                # Write to cache
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

            # Calculate personal bankrupcy ratio per region based on the credit score distribution of that region
            personal_state.personal_bankruptcy[region] = self._calc_personal_bankruptcy(
                region, spot_credit_mean_r
            )

        # Calculate expected reduction from demand based on people's current spending
        demand_reduction = self._calc_demand_reduction(
            personal_state.spot_expense_by_sector
        )
        personal_state.demand_reduction = demand_reduction

        # Assign the result container to SimulateState
        state.personal_state = personal_state

    def _calc_spot_earning(
        self, r: Region, s: Sector, d: Decile, utilisations: Utilisations
    ) -> float:
        spot_earning = 0
        for worker_state in WorkerState:
            spot_earning += (
                utilisations[r, s][worker_state]
                * self._cache_earnings_by_worker_state[(r, s, d, worker_state)]
            )
        return spot_earning

    def _calc_spot_expense_by_sector(
        self,
        region: Region,
        employed_sector: Sector,
        decile: Decile,
        spot_earning: float,
        fear_factor: float,
        utilisations: Utilisations,
    ) -> Mapping[Sector, float]:
        spot_earning_ratio = min(
            spot_earning / self.earnings[(region, employed_sector, decile)], 1.0
        )
        expense_dict = {}
        for expense_sector in Sector:
            expense = (
                self.detailed_expenses[
                    (region, employed_sector, decile, expense_sector)
                ]
                * spot_earning_ratio
                * (1 - fear_factor)
            )
            expense = max(
                expense,
                self.detailed_min_expenses[
                    (region, employed_sector, decile, expense_sector)
                ],
            ) * (1 - utilisations[region, employed_sector][WorkerState.DEAD])
            expense_dict[expense_sector] = expense

        return expense_dict

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
            ppb += self._cache_mixture_weights[(r, s, d)] * self._get_cdf(
                self.default_th, spot_credit_mean[(s, d)], self.credit_std[r]
            )

        return ppb

    def _get_cdf(self, v: float, mu: float, std: float) -> float:
        z_score = max(min(round_to_half_int(v - mu / std), 3), -3)
        try:
            return self._cache_cdf[z_score]
        except KeyError:
            cdf = scipy.stats.norm.cdf(v, mu, std)
            self._cache_cdf[z_score] = cdf
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
                / self._cache_expenses_by_expense_sector[expense_sector]
                if self._cache_expenses_by_expense_sector[expense_sector] > 0
                else 0.0
            )
            for expense_sector in Sector
        }

        return demand_reduction


def round_to_half_int(number: float):
    """Round a number to the closest half integer."""

    return round(number * 2) / 2
