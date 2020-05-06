from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import (
    Mapping,
    Tuple,
    MutableMapping,
    Any,
    Optional,
    Union,
    Generator,
)

import numpy as np
from scipy.optimize import OptimizeResult

from adapter_covid19.enums import (
    LabourState,
    Region,
    Sector,
    Age,
    BusinessSize,
    PrimaryInput,
    FinalUse,
    Decile,
    WorkerState,
)


@dataclass
class SimulateState:
    # Necessary for simulation
    time: int
    lockdown: bool  # TODO: remove
    utilisations: Mapping[Tuple[LabourState, Region, Sector, Age], float]

    # Internal state
    gdp_state: Optional[Union[GdpState, IoGdpState]] = None
    corporate_state: Optional[CorporateState] = None
    personal_state: Optional[PersonalState] = None
    previous: Optional[SimulateState] = None


@dataclass
class InitialiseState:
    economics_kwargs: Mapping[str, Any] = field(default_factory=dict)
    gdp_kwargs: Mapping[str, Any] = field(default_factory=dict)
    personal_kwargs: Mapping[str, Any] = field(default_factory=dict)
    corporate_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class CorporateState:
    gdp_discount_factor: Mapping[Sector, float]
    cash_buffer: Mapping[BusinessSize, Mapping[Sector, np.array]]
    proportion_solvent: Mapping[BusinessSize, Mapping[Sector, float]]
    capital: Mapping[Sector, float] = field(
        default_factory=lambda: {s: 1.0 for s in Sector}
    )


@dataclass
class GdpState:
    gdp: Mapping[Tuple[Region, Sector, Age], float] = field(default_factory=dict)
    workers: Mapping[Tuple[Region, Sector, Age], float] = field(default_factory=dict)
    growth_factor: Mapping[Sector, float] = field(default_factory=dict)
    max_gdp: float = 0
    max_workers: float = 0

    def fraction_gdp_by_sector(self) -> Mapping[Sector, float]:
        return {
            s: sum(
                self.gdp[r, s, a] / self.max_gdp
                for r, a in itertools.product(Region, Age)
            )
            for s in Sector
        }


@dataclass
class IoGdpState(GdpState):
    primary_inputs: Mapping[Tuple[PrimaryInput, Region, Sector, Age], float] = field(
        default_factory=dict
    )
    final_uses: Mapping[Tuple[FinalUse, Sector], float] = field(default_factory=dict)
    compensation_paid: Mapping[Tuple[Region, Sector, Age], float] = field(
        default_factory=dict
    )
    compensation_received: Mapping[Tuple[Region, Sector, Age], float] = field(
        default_factory=dict
    )
    compensation_subsidy: Mapping[Tuple[Region, Sector, Age], float] = field(
        default_factory=dict
    )
    max_primary_inputs: Mapping[
        Tuple[PrimaryInput, Region, Sector, Age], float
    ] = field(default_factory=dict)
    max_final_uses: Mapping[Tuple[FinalUse, Sector], float] = field(
        default_factory=dict
    )
    max_compensation_paid: Mapping[Tuple[Region, Sector, Age], float] = field(
        default_factory=dict
    )
    max_compensation_received: Mapping[Tuple[Region, Sector, Age], float] = field(
        default_factory=dict
    )
    max_compensation_subsidy: Mapping[Tuple[Region, Sector, Age], float] = field(
        default_factory=dict
    )
    _optimise_result: Optional[OptimizeResult] = None

    @property
    def net_operating_surplus(self):
        return {
            s: -sum(
                [
                    self.primary_inputs[PrimaryInput.NET_OPERATING_SURPLUS, r, s, a]
                    for r, a in itertools.product(Region, Age)
                ]
            )
            for s in Sector
        }


# TODO: deprecate in favour of GdpState
@dataclass
class GdpResult:
    gdp: MutableMapping[int, Mapping[Tuple[Region, Sector, Age], float]] = field(
        default_factory=dict
    )
    workers: MutableMapping[int, Mapping[Tuple[Region, Sector, Age], float]] = field(
        default_factory=dict
    )
    growth_factor: MutableMapping[int, Mapping[Sector, float]] = field(
        default_factory=dict
    )
    max_gdp: float = 0
    max_workers: float = 0

    def fraction_gdp_by_sector(self, time: int) -> Mapping[Sector, float]:
        return {
            s: sum(
                self.gdp[time][r, s, a] / self.max_gdp
                for r, a in itertools.product(Region, Age)
            )
            for s in Sector
        }


# TODO: deprecate in favour of IoGdpState
@dataclass
class IoGdpResult(GdpResult):
    primary_inputs: MutableMapping[
        int, Mapping[Tuple[PrimaryInput, Region, Sector, Age], float]
    ] = field(default_factory=dict)
    final_uses: MutableMapping[int, Mapping[Tuple[FinalUse, Sector], float]] = field(
        default_factory=dict
    )
    compensation_paid: MutableMapping[
        int, Mapping[Tuple[Region, Sector, Age], float]
    ] = field(default_factory=dict)
    compensation_received: MutableMapping[
        int, Mapping[Tuple[Region, Sector, Age], float]
    ] = field(default_factory=dict)
    compensation_subsidy: MutableMapping[
        int, Mapping[Tuple[Region, Sector, Age], float]
    ] = field(default_factory=dict)
    max_primary_inputs: Mapping[
        Tuple[PrimaryInput, Region, Sector, Age], float
    ] = field(default_factory=dict)
    max_final_uses: Mapping[Tuple[FinalUse, Sector], float] = field(
        default_factory=dict
    )
    max_compensation_paid: Mapping[Tuple[Region, Sector, Age], float] = field(
        default_factory=dict
    )
    max_compensation_received: Mapping[Tuple[Region, Sector, Age], float] = field(
        default_factory=dict
    )
    max_compensation_subsidy: Mapping[Tuple[Region, Sector, Age], float] = field(
        default_factory=dict
    )

    def update(self, time: int, other: IoGdpState):
        self.gdp[time] = other.gdp
        self.workers[time] = other.workers
        self.growth_factor[time] = other.growth_factor
        self.max_gdp = other.max_gdp
        self.max_workers = other.max_workers
        self.primary_inputs[time] = other.primary_inputs
        self.final_uses[time] = other.final_uses
        self.compensation_paid[time] = other.compensation_paid
        self.compensation_received[time] = other.compensation_received
        self.compensation_subsidy[time] = other.compensation_subsidy
        self.max_primary_inputs = other.max_primary_inputs
        self.max_final_uses = other.max_final_uses
        self.max_compensation_paid = other.max_compensation_paid
        self.max_compensation_received = other.max_compensation_received
        self.max_compensation_subsidy = other.max_compensation_subsidy


@dataclass
class PersonalState:
    time: int
    delta_balance: Mapping[Tuple[Region, Sector, Decile], float]
    balance: Mapping[Tuple[Region, Sector, Decile], float]
    credit_mean: Mapping[Tuple[Region, Sector, Decile], float]
    credit_std: Mapping[Region, float]
    utilisation: Mapping[Tuple[Region, Sector, Decile], float]
    min_expense_cut: Mapping[Tuple[Region, Sector, Decile], float]
    personal_bankruptcy: Mapping[Region, float]


# TODO: Deprecate
@dataclass
class PersonalStateToDeprecate:
    time: int
    delta_balance: Mapping[Sector, Mapping[Decile, float]]
    balance: Mapping[Sector, Mapping[Decile, float]]
    credit_mean: Mapping[Sector, Mapping[Decile, float]]
    credit_std: float
    utilisation: Mapping[Sector, Mapping[LabourState, float]]
    min_expense_cut: Mapping[Sector, Mapping[Decile, float]]
    personal_bankruptcy: float


class Utilisations:
    def __init__(
        self,
        utilisations: Mapping[Tuple[Region, Sector, Age], Mapping[WorkerState, float]],
        worker_data: Mapping[Tuple[Region, Sector, Age], float],
    ):
        self._utilisations = utilisations
        self._workers_by_sector = {
            (r, s, a): worker_data[r, s, a]
            / sum(worker_data[rr, s, aa] for rr, aa in itertools.product(Region, Age))
            for r, s, a in itertools.product(Region, Sector, Age)
        }

    @staticmethod
    def _sum(
        mapping: Generator[Union[Utilisation, Mapping[WorkerState, float]]]
    ) -> Utilisation:
        result = {w: 0 for w in WorkerState}
        for s in mapping:
            for w in WorkerState:
                result[w] += s[w]
        return Utilisation.from_lambdas(result)

    def __getitem__(self, item):
        if isinstance(item, Sector):
            return self._sum(
                {
                    w: self._utilisations[r, item, a][w]
                    * self._workers_by_sector[r, item, a]
                    for w in WorkerState
                }
                for r, a in itertools.product(Region, Age)
            )
        else:
            return self._utilisations[item]


class Utilisation:
    def __init__(
        self,
        p_ill: float,
        p_wfh: float,
        p_furloughed: float,
        p_dead: float,
        p_unemployed: float = 0,
    ):
        """

        :param p_ill:
            Proportion of workforce who are not dead, but ill
            (ILL_WORKING + ILL_FURLOUGHED + ILL_UNEMPLOYED) / (1 - DEAD)
            == ILL_WORKING / (ILL_WORKING + HEALTHY_WFH + HEALTHY_WFO)
            == ILL_FURLOUGHED / (ILL_FURLOUGHED + HEALTHY_FURLOUGHED)
            == ILL_UNEMPLOYED / (ILL_UNEMPLOYED + HEALTHY_UNEMPLOYED)
        :param p_wfh:
            Proportion of working workforce who must wfh
            (HEALTHY_WFH + ILL_WFH) / (HEALTHY_WFH + ILL_WFH + HEALTHY_WFO + ILL_WFO)
        :param p_furloughed:
            Proportion of not working workforce who are furloughed
            (HEALTHY_FURLOUGHED + ILL_FURLOUGHED)
            / (HEALTHY_FURLOUGHED + ILL_FURLOUGHED + HEALTHY_UNEMPLOYED + ILL_UNEMPLOYED)
        :param p_dead:
            Proportion of workforce who are dead
            DEAD
        :param p_unemployed:
            Proportion of workforce who are alive but not working
            (HEALTHY_FURLOUGHED + HEALTHY_UNEMPLOYED + ILL_FURLOUGHED + ILL_UNEMPLOYED) / (1 - DEAD)
        """
        assert np.isclose(sum([p_ill, p_wfh, p_furloughed, p_dead, p_unemployed]), 1)
        assert all(
            0 <= x <= 1 for x in [p_ill, p_wfh, p_furloughed, p_dead, p_unemployed]
        )
        self.p_ill = p_ill
        self.p_wfh = p_wfh
        self.p_furloughed = p_furloughed
        self.p_dead = p_dead
        self.p_unemployed = p_unemployed

    def to_lambdas(self):
        p_not_dead = 1 - self.p_dead
        p_not_working = self.p_unemployed * p_not_dead
        p_working = 1 - p_not_dead - p_not_working
        p_healthy_working = (1 - self.p_ill) * p_working
        p_healthy_not_working = (1 - self.p_ill) * p_not_working
        return {
            WorkerState.HEALTHY_WFO: p_healthy_working * (1 - self.p_wfh),
            WorkerState.HEALTHY_WFH: p_healthy_working * self.p_wfh,
            WorkerState.HEALTHY_FURLOUGHED: p_healthy_not_working * self.p_furloughed,
            WorkerState.HEALTHY_UNEMPLOYED: p_healthy_not_working
            * (1 - self.p_furloughed),
            WorkerState.ILL_WORKING: self.p_ill * p_working,
            WorkerState.ILL_FURLOUGHED: self.p_ill * p_not_working * self.p_furloughed,
            WorkerState.ILL_UNEMPLOYED: self.p_ill
            * p_not_working
            * (1 - self.p_furloughed),
            WorkerState.DEAD: self.p_dead,
        }

    @classmethod
    def from_lambdas(cls, lambdas: Mapping[WorkerState, float]) -> Utilisation:
        raise NotImplementedError

    def __getitem__(self, item):
        return self.to_lambdas()[item]
