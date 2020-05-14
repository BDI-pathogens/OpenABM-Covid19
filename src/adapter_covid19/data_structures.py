from __future__ import annotations

import functools
import itertools
import math
import warnings
from dataclasses import dataclass, field, InitVar
from typing import (
    Any,
    Union,
    Generator,
    Type,
)
from typing import Mapping, Tuple, MutableMapping, Optional

import numpy as np
from scipy.optimize import OptimizeResult

from adapter_covid19.constants import START_OF_TIME
from adapter_covid19.datasources import (
    Reader,
    SectorDataSource,
    RegionSectorAgeDataSource,
    DataSource,
)
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
class ModelParams:
    economics_params: Mapping[str, Any] = field(default_factory=dict)
    gdp_params: Mapping[str, Any] = field(default_factory=dict)
    personal_params: Mapping[str, Any] = field(default_factory=dict)
    corporate_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    lockdown_recovery_time: int = 1
    lockdown_exited_time: int = field(default=0, init=False)
    lockdown_start_time: int = 1000
    lockdown_end_time: int = 1000
    furlough_start_time: int = 1000
    furlough_end_time: int = 1000
    simulation_end_time: int = 2000
    new_spending_day: int = 1000
    ccff_day: int = 1000
    loan_guarantee_day: int = 1000
    model_params: ModelParams = ModelParams()
    epidemic_active: bool = True
    # T * percentage. TODO: extend to per sector, region, age
    ill_ratio: Mapping[int, float] = field(default_factory=dict)
    dead_ratio: Mapping[int, float] = field(default_factory=dict)
    # Multiplier factor on time to load the spread data. It can also be set to be below 1.
    # Eg, if spread_model_time_factor = 5, and current time=2, the model will load ill and dead ratio from
    # the spread model at time=10.
    # Eg2, if spread_model_time_factor = 0.5, and current time=5, the model will load ill and dead ratio from
    # the spread model at time=int(5*0.5)=2.
    spread_model_time_factor: float = 1.0

    # Coefs creating fear factor
    fear_factor_coef_lockdown: float = 1.0
    fear_factor_coef_ill: float = 1.0
    fear_factor_coef_dead: float = 1.0

    simulate_states: MutableMapping[int, SimulateState] = field(
        default_factory=dict, init=False
    )

    datasources: Mapping[str, Type[DataSource]] = field(default_factory=dict)
    gdp: Mapping[Tuple[Region, Sector, Age], float] = field(default=None, init=False)
    workers: Mapping[Tuple[Region, Sector, Age], float] = field(
        default=None, init=False
    )
    furloughed: Mapping[Sector, float] = field(default=None, init=False)
    keyworker: Mapping[Sector, float] = field(default=None, init=False)

    _has_been_lockdown: bool = False
    _utilisations: Mapping = field(
        default_factory=dict, init=False
    )  # For tracking / debugging
    is_loaded: bool = False

    def __post_init__(self):
        self.datasources = {
            "gdp": RegionSectorAgeDataSource,
            "workers": RegionSectorAgeDataSource,
            "furloughed": SectorDataSource,
            "keyworker": SectorDataSource,
        }

    def load(self, reader: Reader) -> None:
        for k, v in self.datasources.items():
            self.__setattr__(k, v(k).load(reader))

        if self.epidemic_active:
            if len(self.ill_ratio) == 0 or len(self.dead_ratio) == 0:
                if self.lockdown_start_time >= 1000:
                    lockdown_start = lockdown_end = 0
                else:
                    lockdown_start = self.lockdown_start_time
                    lockdown_end = self.lockdown_end_time
                end = self.simulation_end_time + 1
                file_name = f"spread_model_cache_{lockdown_start}_{lockdown_end}_{end}"
                df = reader.load_pkl(file_name)
                self.ill_ratio = df["ill ratio"]
                self.dead_ratio = df["dead ratio"]

        self.is_loaded = True

    def _pre_simulation_checks(self, time: int, lockdown: bool) -> None:
        if time == START_OF_TIME and lockdown:
            raise ValueError(
                "Economics model requires simulation to be started before lockdown"
            )
        if self.lockdown_exited_time and lockdown:
            raise NotImplementedError(
                "Bankruptcy/insolvency logic for toggling lockdown needs doing"
            )
        if lockdown and not self._has_been_lockdown:
            self._has_been_lockdown = True
        if not self.lockdown_exited_time and self._has_been_lockdown and not lockdown:
            self.lockdown_exited_time = time

    def generate(
        self,
        time: int,
        dead: Mapping[Tuple[Region, Sector, Age], float],
        ill: Mapping[Tuple[Region, Sector, Age], float],
        lockdown: bool,
        furlough: bool,
        reader: Reader,
    ) -> SimulateState:
        self._pre_simulation_checks(time, lockdown)
        simulate_state = self.simulate_states[time] = SimulateState(
            time=time,
            dead=dead,
            ill={
                (e, r, s, a): ill[r, s, a]
                for e, r, s, a in itertools.product(
                    EmploymentState, Region, Sector, Age
                )
            },  # here we assume illness affects all employment states equally
            lockdown=lockdown,
            furlough=furlough,
            new_spending_day=self.new_spending_day,
            ccff_day=self.ccff_day,
            loan_guarantee_day=self.loan_guarantee_day,
            previous=self.simulate_states.get(time - 1),
            reader=reader,
            fear_factor_coef_lockdown=self.fear_factor_coef_lockdown,
            fear_factor_coef_ill=self.fear_factor_coef_ill,
            fear_factor_coef_dead=self.fear_factor_coef_dead,
        )
        return simulate_state

    def get_ill_ratio_dict(
        self, time: int
    ) -> Mapping[Tuple[Region, Sector, Age], float]:
        time_in_spread_model = int(time * self.spread_model_time_factor)
        try:
            return {
                key: self.ill_ratio[time_in_spread_model]
                for key in itertools.product(Region, Sector, Age)
            }
        except KeyError:
            warnings.warn(
                f"Ill ratio at time {time_in_spread_model} is not provided. Returning 0.0"
            )
            return {key: 0.0 for key in itertools.product(Region, Sector, Age)}

    def get_dead_ratio_dict(
        self, time: int
    ) -> Mapping[Tuple[Region, Sector, Age], float]:
        time_in_spread_model = int(time * self.spread_model_time_factor)
        try:
            return {
                key: self.dead_ratio[time_in_spread_model]
                for key in itertools.product(Region, Sector, Age)
            }
        except KeyError:
            warnings.warn(
                f"Dead ratio at time {time_in_spread_model} is not provided. Returning 0.0"
            )
            return {key: 0.0 for key in itertools.product(Region, Sector, Age)}


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

    # Coefs creating fear factor
    fear_factor_coef_lockdown: float
    fear_factor_coef_ill: float
    fear_factor_coef_dead: float

    # Internal state of the model
    utilisations: Optional[Utilisations] = None
    gdp_state: Optional[Union[GdpState, IoGdpState]] = None
    corporate_state: Optional[CorporateState] = None
    personal_state: Optional[PersonalState] = None
    previous: Optional[SimulateState] = None

    # Needed for generating utilisations if not passed in
    reader: InitVar[Optional[Reader]] = None

    def __post_init__(self, reader: Optional[Reader]):
        if self.utilisations is not None:
            return
        if reader is None:
            raise ValueError("Must provide `reader` if `utilisations` is None")
        keyworker = SectorDataSource("keyworker").load(reader)
        self.utilisations = Utilisations(
            {
                (r, s, a): Utilisation(
                    p_dead=self.dead[r, s, a],
                    p_ill_wfh=self.ill[EmploymentState.WFH, r, s, a],
                    p_ill_wfo=self.ill[EmploymentState.WFO, r, s, a],
                    p_ill_furloughed=self.ill[EmploymentState.FURLOUGHED, r, s, a],
                    p_ill_unemployed=self.ill[EmploymentState.UNEMPLOYED, r, s, a],
                    # keyworker state determines who is constrained to WFH
                    p_wfh=1.0 - keyworker[s] if self.lockdown else 0.0,
                    # if furloughing is available, everybody will be furloughed
                    p_furloughed=float(self.furlough),
                    # this will be an output of the GDP model and overridden accordingly
                    p_not_employed=0.0,
                )
                for r, s, a in itertools.product(Region, Sector, Age)
            },
            reader=reader,
        )

    def get_fear_factor(self) -> float:
        # TODO: maybe there's a more efficient way to do this?
        avg_ill = np.mean(list(self.ill.values()))
        if self.previous is None:
            delta_avg_dead = 0.0
        else:
            delta_avg_dead = np.mean(
                list(self.dead.values())
            )- np.mean(list(self.previous.dead.values()))
        logistic_input = (
            self.fear_factor_coef_lockdown * float(self.lockdown)
            + self.fear_factor_coef_ill * avg_ill
            + self.fear_factor_coef_dead * delta_avg_dead
        )
        logistic_output = 1 / (1 + math.exp(-logistic_input))

        fear_factor = max(logistic_output - 0.5, 0) * 2
        return fear_factor


@dataclass
class CorporateState:
    capital_discount_factor: Mapping[Sector, float]
    proportion_solvent: Mapping[BusinessSize, Mapping[Sector, float]]
    proportion_employees_job_exists: Mapping[Sector, float]


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

    def workers_in_sector(self, s: Sector):
        return np.sum(
            [self.workers[r, s, a] for r, a in itertools.product(Region, Age)]
        )


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
            s: sum(
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
    spot_earning: MutableMapping[Tuple[Region, Sector, Decile], float]
    spot_expense: MutableMapping[Tuple[Region, Sector, Decile], float]
    spot_expense_by_sector: MutableMapping[Tuple[Region, Sector, Decile, Sector], float]
    delta_balance: MutableMapping[Tuple[Region, Sector, Decile], float]
    balance: MutableMapping[Tuple[Region, Sector, Decile], float]
    credit_mean: MutableMapping[Tuple[Region, Sector, Decile], float]
    credit_std: MutableMapping[Region, float]
    personal_bankruptcy: MutableMapping[Region, float]
    demand_reduction: Mapping[Sector, float]


class Utilisations:
    def __init__(
        self,
        utilisations: Mapping[Tuple[Region, Sector, Age], Utilisation],
        worker_data: Optional[Mapping[Tuple[Region, Sector, Age], float]] = None,
        reader: Optional[Reader] = None,
    ):
        if worker_data is None and reader is None:
            raise ValueError("must supply one of `worker_data`, `reader`")
        self._utilisations = utilisations
        if worker_data is None:
            worker_data = RegionSectorAgeDataSource("workers").load(reader)
        self._workers_by_region_sector = {
            (r, s, a): worker_data[r, s, a] / sum(worker_data[r, s, aa] for aa in Age)
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        self._workers_by_sector = {
            (r, s, a): worker_data[r, s, a]
            / sum(worker_data[rr, s, aa] for rr, aa in itertools.product(Region, Age))
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        self._utilisations_by_region_sector: Optional[
            Mapping[Tuple[Region, Sector], Utilisation]
        ] = None
        self._utilisations_by_sector: Optional[Mapping[Sector, Utilisation]] = None
        for u in utilisations.values():
            u.set_container(self)

    def invalidate(self):
        self._utilisations_by_region_sector = None
        self._utilisations_by_sector = None
        self.__getitem__.cache_clear()

    def _calc_utilisations_by_region_sector(self):
        self._utilisations_by_region_sector = {
            (r, s): self._sum(
                {
                    w: self._utilisations[r, s, a][w]
                    * self._workers_by_region_sector[r, s, a]
                    for w in WorkerState
                }
                for a in Age
            )
            for r, s in itertools.product(Region, Sector)
        }

    def _calc_utilisations_by_sector(self):
        self._utilisations_by_sector = {
            s: self._sum(
                {
                    w: self._utilisations[r, s, a][w] * self._workers_by_sector[r, s, a]
                    for w in WorkerState
                }
                for r, a in itertools.product(Region, Age)
            )
            for s in Sector
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

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, item):
        if isinstance(item, Sector):
            if self._utilisations_by_sector is None:
                self._calc_utilisations_by_sector()
            return self._utilisations_by_sector[item]
        elif (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], Region)
            and isinstance(item[1], Sector)
        ):
            if self._utilisations_by_region_sector is None:
                self._calc_utilisations_by_region_sector()
            return self._utilisations_by_region_sector[item]
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
        # TODO: This is simply awful, make caching and cache invalidation better
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
        self._p_ill_wfo = p_ill_wfo
        self._p_ill_wfh = p_ill_wfh
        self._p_ill_furloughed = p_ill_furloughed
        self._p_ill_unemployed = p_ill_unemployed
        self._p_wfh = p_wfh
        self._p_furloughed = p_furloughed
        self._p_dead = p_dead
        self._p_not_employed = p_not_employed
        self._lambdas = None
        self._container = None

    @property
    def p_ill_wfo(self):
        return self._p_ill_wfo

    @property
    def p_ill_wfh(self):
        return self._p_ill_wfh

    @property
    def p_ill_furloughed(self):
        return self._p_ill_furloughed

    @property
    def p_ill_unemployed(self):
        return self._p_ill_unemployed

    @property
    def p_wfh(self):
        return self._p_wfh

    @property
    def p_furloughed(self):
        return self._p_furloughed

    @property
    def p_dead(self):
        return self._p_dead

    @property
    def p_not_employed(self):
        return self._p_not_employed

    @p_ill_wfo.setter
    def p_ill_wfo(self, x: float):
        self._invalidate()
        self._p_ill_wfo = x

    @p_ill_wfh.setter
    def p_ill_wfh(self, x: float):
        self._invalidate()
        self._p_ill_wfh = x

    @p_ill_furloughed.setter
    def p_ill_furloughed(self, x: float):
        self._invalidate()
        self._p_ill_furloughed = x

    @p_ill_unemployed.setter
    def p_ill_unemployed(self, x: float):
        self._invalidate()
        self._p_ill_unemployed = x

    @p_wfh.setter
    def p_wfh(self, x: float):
        self._invalidate()
        self._p_wfh = x

    @p_furloughed.setter
    def p_furloughed(self, x: float):
        self._invalidate()
        self._p_furloughed = x

    @p_dead.setter
    def p_dead(self, x: float):
        self._invalidate()
        self._p_dead = x

    @p_not_employed.setter
    def p_not_employed(self, x: float):
        self._invalidate()
        self._p_not_employed = x

    def _invalidate(self):
        self._lambdas = None
        if self._container is not None:
            self._container.invalidate()

    def set_container(self, container: Utilisations):
        if self._container is not None:
            raise ValueError("Utilisations container already exists")
        self._container = container

    def to_lambdas(self):
        if self._lambdas is None:
            lambda_not_dead = 1 - self.p_dead
            # in the below, being "not employed" implies being alive
            lambda_not_employed = self.p_not_employed * lambda_not_dead
            lambda_furloughed = self.p_furloughed * lambda_not_employed
            lambda_unemployed = lambda_not_employed - lambda_furloughed
            lambda_employed = lambda_not_dead - lambda_not_employed
            lambda_wfh = self.p_wfh * lambda_employed
            lambda_wfo = (1 - self.p_wfh) * lambda_employed
            self._lambdas = {
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
        return self._lambdas

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

        # Beware np.float64s not throwing ZeroDivisionErrors
        try:
            p_ill_wfo = float(lambdas[WorkerState.ILL_WFO]) / float(
                lambdas[WorkerState.HEALTHY_WFO] + lambdas[WorkerState.ILL_WFO]
            )
        except ZeroDivisionError:
            p_ill_wfo = default_values.get(WorkerStateConditional.ILL_WFO, p_ill)

        try:
            p_ill_wfh = float(lambdas[WorkerState.ILL_WFH]) / float(
                lambdas[WorkerState.HEALTHY_WFH] + lambdas[WorkerState.ILL_WFH]
            )
        except ZeroDivisionError:
            p_ill_wfh = default_values.get(WorkerStateConditional.ILL_WFH, p_ill)

        try:
            p_ill_furloughed = float(lambdas[WorkerState.ILL_FURLOUGHED]) / float(
                lambdas[WorkerState.HEALTHY_FURLOUGHED]
                + lambdas[WorkerState.ILL_FURLOUGHED]
            )
        except ZeroDivisionError:
            p_ill_furloughed = default_values.get(
                WorkerStateConditional.ILL_FURLOUGHED, p_ill
            )

        try:
            p_ill_unemployed = float(lambdas[WorkerState.ILL_UNEMPLOYED]) / float(
                lambdas[WorkerState.HEALTHY_UNEMPLOYED]
                + lambdas[WorkerState.ILL_UNEMPLOYED]
            )
        except ZeroDivisionError:
            p_ill_unemployed = default_values.get(
                WorkerStateConditional.ILL_UNEMPLOYED, p_ill
            )

        try:
            p_wfh = float(
                lambdas[WorkerState.HEALTHY_WFH] + lambdas[WorkerState.ILL_WFH]
            ) / float(
                lambdas[WorkerState.HEALTHY_WFH]
                + lambdas[WorkerState.ILL_WFH]
                + lambdas[WorkerState.HEALTHY_WFO]
                + lambdas[WorkerState.ILL_WFO]
            )
        except ZeroDivisionError:
            p_wfh = default_values[WorkerStateConditional.WFH]

        try:
            p_furloughed = float(
                lambdas[WorkerState.HEALTHY_FURLOUGHED]
                + lambdas[WorkerState.ILL_FURLOUGHED]
            ) / float(
                lambdas[WorkerState.HEALTHY_FURLOUGHED]
                + lambdas[WorkerState.ILL_FURLOUGHED]
                + lambdas[WorkerState.HEALTHY_UNEMPLOYED]
                + lambdas[WorkerState.ILL_UNEMPLOYED]
            )
        except ZeroDivisionError:
            p_furloughed = default_values[WorkerStateConditional.FURLOUGHED]

        try:
            p_not_employed = float(
                lambdas[WorkerState.HEALTHY_FURLOUGHED]
                + lambdas[WorkerState.ILL_FURLOUGHED]
                + lambdas[WorkerState.HEALTHY_UNEMPLOYED]
                + lambdas[WorkerState.ILL_UNEMPLOYED]
            ) / float(1 - lambdas[WorkerState.DEAD])
        except ZeroDivisionError:
            p_not_employed = default_values[WorkerStateConditional.NOT_EMPLOYED]

        return cls(
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
