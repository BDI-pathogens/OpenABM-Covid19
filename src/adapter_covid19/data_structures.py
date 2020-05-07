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
    EmploymentState,
    WorkerState,
    WorkerStateConditional,
)


@dataclass
class SimulateState:  # at one point in time
    # Exogenous inputs to the economic model including
    # - state from epidemic model
    # - interventions
    time: int
    # health state
    dead: Mapping[Tuple[Region, Sector, Age], float]
    ill: Mapping[Tuple[EmploymentState, Region, Sector, Age], float]
    # lockdown intervention
    lockdown: bool  # TODO: should reflect more granularly who are key workers
    # furlough intervention
    furlough: bool
    # corporate solvency interventions
    new_spending_day: int
    ccff_day: int
    loan_guarantee_day: int

    # Internal state of the model
    utilisations: Optional[Utilisations] = None
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
    capital_discount_factor: Mapping[Sector, float]
    cash_buffer: Mapping[BusinessSize, Mapping[Sector, np.array]]
    proportion_solvent: Mapping[BusinessSize, Mapping[Sector, float]]


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
    spot_earning: Mapping[Tuple[Region, Sector, Decile], float]
    spot_expense: Mapping[Tuple[Region, Sector, Decile], float]
    spot_expense_by_sector: Mapping[Tuple[Region, Sector, Decile, Sector], float]
    delta_balance: Mapping[Tuple[Region, Sector, Decile], float]
    balance: Mapping[Tuple[Region, Sector, Decile], float]
    credit_mean: Mapping[Tuple[Region, Sector, Decile], float]
    credit_std: Mapping[Region, float]
    utilisation: Mapping[Tuple[Region, Sector, Decile], float]
    personal_bankruptcy: Mapping[Region, float]
    demand_reduction: Mapping[Sector, float]


class Utilisations:
    def __init__(
        self,
        utilisations: Mapping[Tuple[Region, Sector, Age], Utilisation],
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
    ) -> Mapping[WorkerState, float]:
        result = {w: 0 for w in WorkerState}
        for s in mapping:
            for w in WorkerState:
                result[w] += s[w]
        return result

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
        elif (
            isinstance(item, tuple)
            and len(item) == 4
            and isinstance(item[0], LabourState)
        ):
            # TODO: deprecate
            # convenience function for legacy models expecting old dictionary interface
            l, r, s, a = item
            u = self._utilisations[r, s, a].to_lambdas()
            if l == LabourState.WORKING:
                return u[WorkerState.HEALTHY_WFO]
            elif l == LabourState.WFH:
                return u[WorkerState.HEALTHY_WFH]
            elif l == LabourState.ILL:
                return u[WorkerState.ILL_WFH] + u[WorkerState.ILL_WFO]
            elif l == LabourState.FURLOUGHED:
                return u[WorkerState.ILL_FURLOUGHED] + u[WorkerState.HEALTHY_FURLOUGHED]
            elif l == LabourState.UNEMPLOYED:
                return (
                    u[WorkerState.ILL_UNEMPLOYED]
                    + u[WorkerState.HEALTHY_UNEMPLOYED]
                    + u[WorkerState.DEAD]
                )
        else:
            return self._utilisations[item]


class Utilisation:
    def __init__(
        self,
        p_dead: float,
        p_ill_wfo: float,
        p_ill_wfh: float,
        p_ill_furloughed: float,
        p_ill_unemployed: float,
        p_wfh: float,
        p_furloughed: float,
        p_not_employed: float = 0,
    ):
        """

        :param p_dead:
            Proportion of workforce who are dead
            p_dead == DEAD
        :param p_ill_wfo:
            Proportion of workforce who are employed and are not constrained to work from home, but have fallen ill
            p_ill_wfo == ILL_WFO / (HEALTHY_WFO + ILL_WFO)
        :param p_ill_wfh:
            Proportion of workforce who are employed and are constrained to work from home, but have fallen ill
            p_ill_wfh == ILL_WFH / (HEALTHY_WFH + ILL_WFH)
        :param p_ill_furloughed:
            Proportion of workforce who are furloughed, who have fallen ill
            p_ill_furloughed == ILL_FURLOUGHED / (HEALTHY_FURLOUGHED + ILL_FURLOUGHED)
        :param p_ill_unemployed:
            Proportion of workforce who are unemployed, who have fallen ill
            p_ill_unemployed == ILL_UNEMPLOYED / (HEALTHY_UNEMPLOYED + ILL_UNEMPLOYED)
        :param p_wfh:
            Proportion of working workforce who must wfh
            p_wfh == (HEALTHY_WFH + ILL_WFH) / (HEALTHY_WFH + ILL_WFH + HEALTHY_WFO + ILL_WFO)
        :param p_furloughed:
            Proportion of not working workforce who are furloughed
            p_furloughed == (HEALTHY_FURLOUGHED + ILL_FURLOUGHED)
                            / (HEALTHY_FURLOUGHED + ILL_FURLOUGHED + HEALTHY_UNEMPLOYED + ILL_UNEMPLOYED)
        :param p_not_employed:
            Proportion of workforce who are alive but not working
            p_not_employed == (HEALTHY_FURLOUGHED + HEALTHY_UNEMPLOYED + ILL_FURLOUGHED + ILL_UNEMPLOYED)
                              / (1 - DEAD)
        """
        assert all(
            0 <= x <= 1
            for x in [
                p_ill_wfo,
                p_ill_wfh,
                p_ill_furloughed,
                p_ill_unemployed,
                p_wfh,
                p_furloughed,
                p_dead,
                p_not_employed,
            ]
        )
        self.p_ill_wfo = p_ill_wfo
        self.p_ill_wfh = p_ill_wfh
        self.p_ill_furloughed = p_ill_furloughed
        self.p_ill_unemployed = p_ill_unemployed
        self.p_wfh = p_wfh
        self.p_furloughed = p_furloughed
        self.p_dead = p_dead
        self.p_not_employed = p_not_employed

    def to_lambdas(self):
        lambda_not_dead = 1 - self.p_dead
        # in the below, being "not employed" implies being alive
        lambda_not_employed = self.p_not_employed * lambda_not_dead
        lambda_furloughed = self.p_furloughed * lambda_not_employed
        lambda_unemployed = lambda_not_employed - lambda_furloughed
        lambda_employed = lambda_not_dead - lambda_not_employed
        lambda_wfh = self.p_wfh * lambda_employed
        lambda_wfo = (1 - self.p_wfh) * lambda_employed
        return {
            WorkerState.HEALTHY_WFO: (1 - self.p_ill_wfo) * lambda_wfo,
            WorkerState.HEALTHY_WFH: (1 - self.p_ill_wfh) * lambda_wfh,
            WorkerState.HEALTHY_FURLOUGHED: (1 - self.p_ill_furloughed)
            * lambda_furloughed,
            WorkerState.HEALTHY_UNEMPLOYED: (1 - self.p_ill_unemployed)
            * lambda_unemployed,
            WorkerState.ILL_WFO: self.p_ill_wfo * lambda_wfo,
            WorkerState.ILL_WFH: self.p_ill_wfh * lambda_wfh,
            WorkerState.ILL_FURLOUGHED: self.p_ill_furloughed * lambda_furloughed,
            WorkerState.ILL_UNEMPLOYED: self.p_ill_unemployed * lambda_unemployed,
            WorkerState.DEAD: self.p_dead,
        }

    def to_dict(self):
        return {
            WorkerStateConditional.DEAD: self.p_dead,
            WorkerStateConditional.ILL_WFO: self.p_ill_wfo,
            WorkerStateConditional.ILL_WFH: self.p_ill_wfh,
            WorkerStateConditional.ILL_FURLOUGHED: self.p_ill_furloughed,
            WorkerStateConditional.ILL_UNEMPLOYED: self.p_ill_unemployed,
            WorkerStateConditional.WFH: self.p_wfh,
            WorkerStateConditional.FURLOUGHED: self.p_furloughed,
            WorkerStateConditional.NOT_EMPLOYED: self.p_not_employed
        }

    @classmethod
    def from_lambdas(
        cls,
        lambdas: Mapping[WorkerState, float],
        default_values: Optional[Mapping[WorkerStateConditional, float]] = None,
    ) -> Utilisation:
        default_values = {} if default_values is None else default_values

        p_dead = lambdas[WorkerState.DEAD]
        assert (
            p_dead != 1.0
        ), "conversion form lambdas only well-defined if there are survivors in the population"

        p_ill = (
            lambdas[WorkerState.ILL_WFO]
            + lambdas[WorkerState.ILL_WFH]
            + lambdas[WorkerState.ILL_FURLOUGHED]
            + lambdas[WorkerState.ILL_UNEMPLOYED]
        ) / (1 - p_dead)

        try:
            p_ill_wfo = lambdas[WorkerState.ILL_WFO] / (
                lambdas[WorkerState.HEALTHY_WFO] + lambdas[WorkerState.ILL_WFO]
            )
        except ZeroDivisionError:
            p_ill_wfo = default_values.get(WorkerStateConditional.ILL_WFO, p_ill)

        try:
            p_ill_wfh = lambdas[WorkerState.ILL_WFH] / (
                lambdas[WorkerState.HEALTHY_WFH] + lambdas[WorkerState.ILL_WFH]
            )
        except ZeroDivisionError:
            p_ill_wfh = default_values.get(WorkerStateConditional.ILL_WFH, p_ill)

        try:
            p_ill_furloughed = lambdas[WorkerState.ILL_FURLOUGHED] / (
                lambdas[WorkerState.HEALTHY_FURLOUGHED]
                + lambdas[WorkerState.ILL_FURLOUGHED]
            )
        except ZeroDivisionError:
            p_ill_furloughed = default_values.get(
                WorkerStateConditional.ILL_FURLOUGHED, p_ill
            )

        try:
            p_ill_unemployed = lambdas[WorkerState.ILL_UNEMPLOYED] / (
                lambdas[WorkerState.HEALTHY_UNEMPLOYED]
                + lambdas[WorkerState.ILL_UNEMPLOYED]
            )
        except ZeroDivisionError:
            p_ill_unemployed = default_values.get(
                WorkerStateConditional.ILL_UNEMPLOYED, p_ill
            )

        try:
            p_wfh = (
                lambdas[WorkerState.HEALTHY_WFH] + lambdas[WorkerState.ILL_WFH]
            ) / (
                lambdas[WorkerState.HEALTHY_WFH]
                + lambdas[WorkerState.ILL_WFH]
                + lambdas[WorkerState.HEALTHY_WFO]
                + lambdas[WorkerState.ILL_WFO]
            )
        except ZeroDivisionError:
            p_wfh = default_values[WorkerStateConditional.WFH]

        try:
            p_furloughed = (
                (
                    lambdas[WorkerState.HEALTHY_FURLOUGHED]
                    + lambdas[WorkerState.ILL_FURLOUGHED]
                )
                / (
                    lambdas[WorkerState.HEALTHY_FURLOUGHED]
                    + lambdas[WorkerState.ILL_FURLOUGHED]
                    + lambdas[WorkerState.HEALTHY_UNEMPLOYED]
                    + lambdas[WorkerState.ILL_UNEMPLOYED]
                ),
            )
        except ZeroDivisionError:
            p_furloughed = default_values[WorkerStateConditional.FURLOUGHED]

        try:
            p_not_employed = (
                lambdas[WorkerState.HEALTHY_FURLOUGHED]
                + lambdas[WorkerState.ILL_FURLOUGHED]
                + lambdas[WorkerState.HEALTHY_UNEMPLOYED]
                + lambdas[WorkerState.ILL_UNEMPLOYED]
            ) / (1 - lambdas[WorkerState.DEAD])
        except ZeroDivisionError:
            p_not_employed = default_values[WorkerStateConditional.NOT_EMPLOYED]

        return Utilisation(
            p_dead=p_dead,
            p_ill_wfo=p_ill_wfo,
            p_ill_wfh=p_ill_wfh,
            p_ill_furloughed=p_ill_furloughed,
            p_ill_unemployed=p_ill_unemployed,
            p_wfh=p_wfh,
            p_furloughed=p_furloughed,
            p_not_employed=p_not_employed,
        )

    def __getitem__(self, item):
        return self.to_lambdas()[item]

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        return all(np.isclose(self_dict[key], other_dict[key]) for key in self_dict)


if __name__ == "__main__":
    u = Utilisation(
        p_dead=0.0001,
        p_ill_wfo=0.01,
        p_ill_wfh=0.01,
        p_ill_furloughed=0.01,
        p_ill_unemployed=0.01,
        p_wfh=0.7,
        p_furloughed=0.8,
        p_not_employed=0.5,
    )
    print(u.to_dict())
    l = u.to_lambdas()
    print(l)
    u2 = Utilisation.from_lambdas(l)
    print(u2.to_dict())
