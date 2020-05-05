from __future__ import annotations

import abc
import copy
import itertools
import logging
from dataclasses import dataclass, field
from typing import Tuple, Mapping, MutableMapping, Sequence, Optional, Union, List

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from adapter_covid19.constants import START_OF_TIME, DAYS_IN_A_YEAR
from adapter_covid19.datasources import (
    Reader,
    RegionSectorAgeDataSource,
    SectorDataSource,
    WeightMatrix,
    DataSource,
    DataFrameDataSource,
    RegionDataSource,
)
from adapter_covid19.enums import (
    Region,
    Sector,
    Age,
    M,
    PrimaryInput,
    FinalUse,
    LabourState,
)

LOGGER = logging.getLogger(__name__)


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
    max_final_uses: MutableMapping[
        int, Mapping[Tuple[FinalUse, Sector], float]
    ] = field(default_factory=dict)
    max_compensation_paid: MutableMapping[
        int, Mapping[Tuple[Region, Sector, Age], float]
    ] = field(default_factory=dict)
    max_compensation_received: MutableMapping[
        int, Mapping[Tuple[Region, Sector, Age], float]
    ] = field(default_factory=dict)
    max_compensation_subsidy: MutableMapping[
        int, Mapping[Tuple[Region, Sector, Age], float]
    ] = field(default_factory=dict)

    def update(self, other: IoGdpResult):
        self.gdp.update(other.gdp)
        self.workers.update(other.workers)
        self.growth_factor.update(other.growth_factor)
        self.max_gdp = other.max_gdp
        self.max_workers = other.max_workers
        self.primary_inputs.update(other.primary_inputs)
        self.final_uses.update(other.final_uses)
        self.compensation_paid.update(other.compensation_paid)
        self.compensation_received.update(other.compensation_received)
        self.compensation_subsidy.update(other.compensation_subsidy)
        self.max_compensation_paid.update(other.max_compensation_paid)
        self.max_compensation_received.update(other.max_compensation_received)
        self.max_compensation_subsidy.update(other.max_compensation_subsidy)


class BaseGdpModel(abc.ABC):
    gdp: Mapping[Tuple[Region, Sector, Age], float]
    workers: Mapping[Tuple[Region, Sector, Age], float]
    keyworker: Mapping[Sector, float]
    growth_rates: Mapping[Sector, float]

    def __init__(self, **kwargs):
        if kwargs:
            LOGGER.warning(f"Unused kwargs in {self.__class__.__name__}: {kwargs}")
        self.results = GdpResult()
        self.datasources = self._get_datasources()
        for k, v in self.datasources.items():
            self.__setattr__(k, None)

    @property
    def max_gdp(self):
        return self.results.max_gdp

    @property
    def max_workers(self):
        return self.results.max_workers

    @abc.abstractmethod
    def _get_datasources(self) -> Mapping[str, DataSource]:
        # TODO: This should really be a functools.cached_property, but no python 3.8
        datasources = {
            "gdp": RegionSectorAgeDataSource,
            "workers": RegionSectorAgeDataSource,
            "growth_rates": SectorDataSource,
            "keyworker": SectorDataSource,
        }
        return {k: v(k) for k, v in datasources.items()}

    def _check_data(self) -> None:
        """
        Checks that sectors, regions and ages are consistent between data
        :return:
        """
        if "gdp" not in self.datasources:
            raise ValueError("Trying to simulate gdp without gdp...?")
        for k, v in self.datasources.items():
            if isinstance(v, RegionSectorAgeDataSource):
                regions, sectors, ages = [
                    set(x) for x in zip(*list(self.__getattribute__(k).keys()))
                ]
                if regions != set(Region):
                    raise ValueError(f"Inconsistent data for {k}: {regions}, {Region}")
                if sectors != set(Sector):
                    raise ValueError(f"Inconsistent data for {k}: {sectors}, {Sector}")
                if ages != set(Age):
                    raise ValueError(f"Inconsistent data for {k}: {ages}, {Age}")
            elif isinstance(v, SectorDataSource):
                sectors = set(self.__getattribute__(k).keys())
                if sectors != set(Sector):
                    raise ValueError(f"Inconsistent data for {k}: {sectors}, {Sector}")
            elif isinstance(v, RegionDataSource):
                regions = set(self.__getattribute__(k).keys())
                if regions != set(Region):
                    raise ValueError(f"Inconsistent data for {k}: {regions}, {Region}")
            elif isinstance(v, WeightMatrix):
                matrix = self.__getattribute__(k)
                if any(len(Sector) != s for s in matrix.shape):
                    raise ValueError(
                        f"Inconsistent data for {k}: {len(Sector)}, {matrix.shape}"
                    )
            elif isinstance(v, DataFrameDataSource):
                # We don't know what schema the dataframe is, so we can't check it
                pass
            else:
                raise NotImplementedError(
                    f"Data checks not implemented for {v.__class__.__name__}"
                )

    def load(self, reader: Reader):
        for k, v in self.datasources.items():
            self.__setattr__(k, v.load(reader))
        self._check_data()
        self.results.max_gdp = sum(
            self.gdp[key] for key in itertools.product(Region, Sector, Age)
        )
        self.results.max_workers = sum(
            self.workers[key] for key in itertools.product(Region, Sector, Age)
        )

    def _apply_growth_factor(
        self, time: int, lockdown: bool, gdp: Mapping[Tuple[Region, Sector, Age], float]
    ) -> Tuple[Mapping[Sector, float], Mapping[Tuple[Region, Sector, Age], float]]:
        if (
            time - 1 not in self.results.growth_factor
            or not self.results.growth_factor[time - 1]
        ):
            growth_factor = {s: 1 for s in Sector}
        elif lockdown:
            # No growth in lockdown
            growth_factor = copy.deepcopy(self.results.growth_factor[time - 1])
        else:
            growth_factor = {
                s: self.results.growth_factor[time - 1][s]
                * (1 + self.growth_rates.get(s, 0.0) / DAYS_IN_A_YEAR)
                for s in Sector
            }
        return (
            growth_factor,
            {(r, s, a): gdp[(r, s, a)] * growth_factor[s] for (r, s, a) in gdp.keys()},
        )

    @abc.abstractmethod
    def simulate(
        self,
        time: int,
        lockdown: bool,
        utilisations: Mapping[Tuple[LabourState, Region, Sector, Age], float],
        **kwargs,
    ) -> None:
        pass


class LinearGdpModel(BaseGdpModel):
    gdp: Mapping[Tuple[Region, Sector, Age], float]
    workers: Mapping[Tuple[Region, Sector, Age], float]
    wfh: Mapping[str, float]
    vulnerability: Mapping[str, float]

    def _get_datasources(self) -> Mapping[str, DataSource]:
        datasources = {
            "gdp": RegionSectorAgeDataSource,
            "workers": RegionSectorAgeDataSource,
            "growth_rates": SectorDataSource,
            "keyworker": SectorDataSource,
            "vulnerability": SectorDataSource,
            "wfh": SectorDataSource,
        }
        return {k: v(k) for k, v in datasources.items()}

    def _simulate_gdp(
        self, region: Region, sector: Sector, age: Age, utilisation: float
    ) -> float:
        wfh_factor = self.wfh[sector]
        vulnerability_factor = self.vulnerability[sector]
        return (
            wfh_factor + (vulnerability_factor - wfh_factor) * utilisation
        ) * self.gdp[region, sector, age]

    def _simulate_workers(
        self, region: Region, sector: Sector, age: Age, utilisation: float
    ) -> float:
        return utilisation * self.workers[region, sector, age]

    def simulate(
        self,
        time: int,
        lockdown: bool,
        utilisations: Mapping[Tuple[LabourState, Region, Sector, Age], float],
        **kwargs,
    ) -> None:
        # FIXME: hacked to work with old utilisations
        utilisations = {
            (r, s, a): utilisations[LabourState.WORKING, r, s, a]
            for r, s, a in itertools.product(Region, Sector, Age)
        }

        gdp = {
            (r, s, a): self._simulate_gdp(r, s, a, u)
            for (r, s, a), u in utilisations.items()
        }
        workers = {
            (r, s, a): self._simulate_workers(r, s, a, u)
            for (r, s, a), u in utilisations.items()
        }
        growth_factor, gdp = self._apply_growth_factor(time, lockdown, gdp)
        self.results.gdp[time] = gdp
        self.results.growth_factor[time] = growth_factor
        self.results.workers[time] = workers


class SupplyDemandGdpModel(BaseGdpModel):
    gdp: Mapping[Tuple[Region, Sector, Age], float]
    workers: Mapping[Tuple[Region, Sector, Age], float]
    wfh: Mapping[Sector, float]
    vulnerability: Mapping[Sector, float]
    supply: np.array
    demand: np.array

    def __init__(self, theta: float = 1.2, **kwargs):
        super().__init__(**kwargs)
        self.theta = theta

    def _get_datasources(self) -> Mapping[str, DataSource]:
        datasources = {
            "gdp": RegionSectorAgeDataSource,
            "workers": RegionSectorAgeDataSource,
            "growth_rates": SectorDataSource,
            "keyworker": SectorDataSource,
            "vulnerability": SectorDataSource,
            "wfh": SectorDataSource,
            "supply": WeightMatrix,
            "demand": WeightMatrix,
        }
        return {k: v(k) for k, v in datasources.items()}

    def _simulate_gdp(
        self, utilisations: Mapping[Tuple[Region, Sector, Age], float],
    ) -> Mapping[Tuple[Region, Sector, Age], float]:
        """
        Refer to the model docs for documentation of this function
        :param utilisations:
        :return: GDP per region, sector, age
        """
        # Reforumlate the problem in terms of Region, Age, Sector (makes easier to solve)
        gdp = {}
        for region, age in itertools.product(Region, Age):
            lam = np.array([utilisations[region, s, age] for s in Sector])
            n = len(Sector)
            c = -np.array([self.gdp[region, s, age] for s in Sector])
            h = {s: self.wfh[s] for s in Sector}
            H = np.array([self.wfh[s] for s in Sector])
            WY = self.supply
            WD = self.demand
            bounds = [(0, 1) for _ in range(n)]
            y_max = {
                si: sum(WY[i, j] * (1 - h[sj]) for j, sj in enumerate(Sector))
                for i, si in enumerate(Sector)
            }
            d_max = {
                si: sum(WD[i, j] * (1 - h[sj]) for j, sj in enumerate(Sector))
                for i, si in enumerate(Sector)
            }
            alpha_hat = np.array(
                [
                    (1 - h[s] * self.theta) / min(d_max[s], y_max[s])
                    if h[s] != 1
                    else 1
                    for s in Sector
                ]
            )
            alphalam = np.diag(alpha_hat * lam)
            ialphalamwy = np.eye(n) - alphalam.dot(WY)
            ialphalamwd = np.eye(n) - alphalam.dot(WD)
            aub = np.vstack([ialphalamwy, ialphalamwd,])
            bub = np.concatenate(
                [
                    (
                        np.eye(n) - (1 - self.theta) * np.diag(lam) - alphalam.dot(WY)
                    ).dot(H),
                    (
                        np.eye(n) - (1 - self.theta) * np.diag(lam) - alphalam.dot(WD)
                    ).dot(H),
                ]
            )
            if aub[n - 1, -1] == 0:
                aub[n - 1, -1] = 1
                aub[2 * n - 1, -1] = 1
                bub[n - 1] = 1
                bub[2 * n - 1] = 1
            r = linprog(
                c=c,
                A_ub=aub,
                b_ub=bub,
                bounds=bounds,
                x0=None,
                method="revised simplex",
            )
            if not r.success:
                raise ValueError(r.message)
            for x, sector in zip(r.x, Sector):
                gdp[region, sector, age] = x * self.gdp[region, sector, age]
        return gdp

    def _simulate_workers(
        self, region: Region, sector: Sector, age: Age, utilisation: float
    ) -> float:
        return utilisation * self.workers[region, sector, age]

    def simulate(
        self,
        time: int,
        lockdown: bool,
        utilisations: Mapping[Tuple[LabourState, Region, Sector, Age], float],
        **kwargs,
    ) -> None:
        # FIXME: hacked to work with old utilisations
        utilisations = {
            (r, s, a): utilisations[LabourState.WORKING, r, s, a]
            for r, s, a in itertools.product(Region, Sector, Age)
        }

        workers = {
            (r, s, a): self._simulate_workers(r, s, a, u)
            for (r, s, a), u in utilisations.items()
        }
        gdp = self._simulate_gdp(utilisations)
        growth_factor, gdp = self._apply_growth_factor(time, lockdown, gdp)
        self.results.gdp[time] = gdp
        self.results.growth_factor[time] = growth_factor
        self.results.workers[time] = workers


@dataclass
class Bound:
    A_ub: Optional[np.array]
    b_ub: Optional[np.array]
    A_eq: Optional[np.array]
    b_eq: Optional[np.array]


@dataclass
class Bounds:
    A_ub: List[np.array]
    b_ub: List[np.array]
    A_eq: List[np.array]
    b_eq: List[np.array]

    def to_array(self):
        return tuple(
            np.concatenate(x, axis=0)
            for x in [self.A_ub, self.b_ub, self.A_eq, self.b_eq]
        )


class CobbDouglasLPSetup:
    def __init__(self):
        value_adds = [M.L, M.K]
        self.variables = (
            [("q", i) for i in Sector]
            + [("d", i, j) for i in Sector for j in Sector]
            + [("x", m, i) for m in M for i in Sector]
            + [("xtilde", m, i) for m in value_adds for i in Sector]
            + [("y", i) for i in Sector]
        )
        self.xtilde_iot = pd.DataFrame([])
        self.objective_c = np.array([])
        self.objective_per_sector = {}
        self.bounds = Bounds([], [], [], [])
        l_bounds = {
            "q": 0.0,
            "d": 0.0,
            "x": 0.0,
            "xtilde": 0.0,
            ("xtilde", M.K): -np.inf,
            "y": 0.0,
        }
        u_bounds = {
            "q": np.inf,
            "d": np.inf,
            "x": np.inf,
            "xtilde": np.inf,
            "y": np.inf,
        }
        self.lp_bounds = list(
            zip(
                [self._bound_lookup(v, bounds=l_bounds) for v in self.variables],
                [self._bound_lookup(v, bounds=u_bounds) for v in self.variables],
            )
        )

    def _bound_lookup(
        self, v: Tuple, bounds: Mapping[Union[str, Tuple[str, M]], float]
    ) -> float:
        if v in bounds:
            return bounds[v]
        elif v[0:2] in bounds:
            return bounds[v[0:2]]
        elif v[0:1] in bounds:
            return bounds[v[0:1]]
        elif v[0] in bounds:
            return bounds[v[0]]

    def V(self, *args) -> Sequence[int]:
        return self.variables.index(tuple(args))

    def indicator(self, *v) -> np.array:
        if len(v) == 1 and isinstance(v[0], int):
            index = v[0]
        else:
            index = self.V(*v)
        arr = np.zeros(len(self.variables))
        arr[index] = 1.0
        return arr

    def add_constraint(self, bound: Bound, bounds: Bounds) -> Bounds:
        bounds = copy.deepcopy(bounds)
        if bound.A_ub is not None:
            bounds.A_ub.append(bound.A_ub)
            bounds.b_ub.append(bound.b_ub)
        if bound.A_eq is not None:
            bounds.A_eq.append(bound.A_eq)
            bounds.b_eq.append(bound.b_eq)
        return bounds

    def update_constraint(self, bound: Bound) -> None:
        if bound.A_ub is not None:
            self.bounds.A_ub.append(bound.A_ub)
            self.bounds.b_ub.append(bound.b_ub)
        if bound.A_eq is not None:
            self.bounds.A_eq.append(bound.A_eq)
            self.bounds.b_eq.append(bound.b_eq)

    def c_production_function_lin(
        self,
        gamma_d_dict: Mapping[Tuple[Sector, Sector], float],
        gamma_x_dict: Mapping[Tuple[Sector, Sector], float],
        Lambda_dict: Mapping[Sector, float],
        substitution_rate: float,
    ):
        constr_arr = []

        for i in Sector:
            for jprime in Sector:
                # jprime constr
                if gamma_d_dict[jprime, i] > 0.0:
                    constr_arr.append(
                        self.indicator("q", i)
                        - Lambda_dict[i]
                        * (
                            (1 - substitution_rate)
                            * 1
                            / gamma_d_dict[jprime, i]
                            * self.indicator("d", jprime, i)
                            + substitution_rate
                            * (
                                np.sum(
                                    [
                                        gamma_d_dict[j, i] * self.indicator("d", j, i)
                                        for j in Sector
                                    ],
                                    axis=0,
                                )
                                + np.sum(
                                    [
                                        gamma_x_dict[m, i] * self.indicator("x", m, i)
                                        for m in M
                                    ],
                                    axis=0,
                                )
                            )
                        )
                    )
            # mprime constr
            for mprime in M:
                if gamma_x_dict[mprime, i] > 0.0:
                    constr_arr.append(
                        self.indicator("q", i)
                        - Lambda_dict[i]
                        * (
                            (1 - substitution_rate)
                            * 1
                            / gamma_x_dict[mprime, i]
                            * self.indicator("x", mprime, i)
                            + substitution_rate
                            * (
                                np.sum(
                                    [
                                        gamma_d_dict[j, i] * self.indicator("d", j, i)
                                        for j in Sector
                                    ],
                                    axis=0,
                                )
                                + np.sum(
                                    [
                                        gamma_x_dict[m, i] * self.indicator("x", m, i)
                                        for m in M
                                    ],
                                    axis=0,
                                )
                            )
                        )
                    )
        A = np.array(constr_arr)
        self.update_constraint(Bound(A, np.zeros(A.shape[0]), None, None))

    def c_input(self, o_iot: pd.DataFrame, q_iot: pd.DataFrame, p_tau: float):
        factor = 1 / (1 - (o_iot.sum(axis=0) / q_iot) * p_tau)
        A = np.array(
            [
                self.indicator("q", i)
                - (
                    factor.loc[i]
                    * (
                        np.sum([self.indicator("d", j, i) for j in Sector], axis=0)
                        + self.indicator("x", M.I, i)
                        + self.indicator("xtilde", M.L, i)
                        + self.indicator("xtilde", M.K, i)
                    )
                )
                for i in Sector
            ]
        )
        # normalization
        normalization = np.array([1 / q_iot.loc[i] for i in Sector])
        A = np.multiply(A, normalization[:, None])
        self.update_constraint(Bound(None, None, A, np.zeros(A.shape[0])))

    def c_output(self, q_iot: pd.DataFrame):
        A = np.array(
            [
                self.indicator("q", i)
                - (
                    np.sum([self.indicator("d", i, j) for j in Sector], axis=0)
                    + self.indicator("y", i)
                )
                for i in Sector
            ]
        )
        # normalization
        normalization = np.array([1 / q_iot.loc[i] for i in Sector])
        A = np.multiply(A, normalization[:, None])
        self.update_constraint(Bound(None, None, A, np.zeros(A.shape[0])))

    def c_capital(self, p_kappa: Mapping[Sector, float]):
        const = np.array([p_kappa[i] * self.xtilde_iot.loc[M.K, i] for i in Sector])
        A = np.array([self.indicator("x", M.K, i) for i in Sector])
        normalization = np.array([1 / self.xtilde_iot.loc[M.K, i] for i in Sector])
        const = np.multiply(const, normalization)
        A = np.multiply(A, normalization[:, None])
        return Bound(None, None, A, const)

    def c_demand(self, p_delta: pd.DataFrame, ytilde_iot: pd.DataFrame):
        const = np.array(
            [
                np.sum([p_delta.loc[i, u] * ytilde_iot.loc[i, u] for u in FinalUse])
                for i in Sector
            ]
        )
        A = np.array([self.indicator("y", i) for i in Sector])
        normalization = np.array([1 / ytilde_iot.sum(axis=1).loc[i] for i in Sector])
        const = np.multiply(
            const, normalization
        )  # if normalized, should all be 1 if p_delta is 1
        A = np.multiply(A, normalization[:, None])
        self.update_constraint(Bound(A, const, None, None))

    def c_labour_quantity(
        self,
        p_lambda_dict: Mapping[Tuple[Sector, LabourState], float],
        wfh_productivity: Mapping[Sector, float],
    ) -> Bound:
        const = np.array(
            [
                (
                    p_lambda_dict[i, LabourState.WORKING]
                    + wfh_productivity[i] * p_lambda_dict[i, LabourState.WFH]
                )
                * self.xtilde_iot.loc[M.L, i]
                for i in Sector
            ]
        )
        A = np.array([self.indicator("x", M.L, i) for i in Sector])
        normalization = np.array([1 / self.xtilde_iot.loc[M.L, i] for i in Sector])
        const = np.multiply(const, normalization)
        A = np.multiply(A, normalization[:, None])
        return Bound(None, None, A, const)

    def c_labour_compensation(self, p_lambda: pd.DataFrame) -> Bound:
        const = np.array(
            [
                (
                    p_lambda.loc[i, LabourState.WORKING]
                    + p_lambda.loc[i, LabourState.WFH]
                    + p_lambda.loc[i, LabourState.ILL]
                )
                * self.xtilde_iot.loc[M.L, i]
                for i in Sector
            ]
        )
        A = np.array([self.indicator("xtilde", M.L, i) for i in Sector])
        normalization = np.array([1 / self.xtilde_iot.loc[M.L, i] for i in Sector])
        const = np.multiply(const, normalization)
        A = np.multiply(A, normalization[:, None])
        return Bound(None, None, A, const)

    def initial_setup(
        self,
        iot_p: pd.DataFrame,
        dtilde_iot: pd.DataFrame,
        ytilde_iot: pd.DataFrame,
        p_delta: pd.DataFrame,
        p_tau: float,
        substitution_rate: float,
    ) -> None:
        self.iot_p = iot_p
        self.dtilde_iot = dtilde_iot
        self.ytilde_iot = ytilde_iot
        self.xtilde_iot = pd.concat(
            [
                iot_p[PrimaryInput.IMPORTS],
                iot_p[PrimaryInput.COMPENSATION],
                iot_p[
                    [
                        PrimaryInput.FIXED_CAPITAL_CONSUMPTION,
                        PrimaryInput.NET_OPERATING_SURPLUS,
                    ]
                ].sum(axis=1),
            ],
            axis=1,
        ).T
        self.xtilde_iot.index = M
        # x~[M.K, T] == 0, so we add a small epsilon
        self.xtilde_iot = np.maximum(self.xtilde_iot, 1e-6)
        self.ytilde_total_iot = self.ytilde_iot.sum(axis=1)
        self.gamma_d = dtilde_iot.div(
            dtilde_iot.sum(axis=0) + self.xtilde_iot.sum(axis=0)
        )
        self.gamma_x = self.xtilde_iot.div(
            dtilde_iot.sum(axis=0) + self.xtilde_iot.sum(axis=0)
        )

        self.o_iot = iot_p[
            [PrimaryInput.TAXES_PRODUCTION, PrimaryInput.TAXES_PRODUCTS]
        ].T
        self.q_iot = dtilde_iot.sum(axis=0) + iot_p.sum(axis=1)
        assert np.allclose(
            (dtilde_iot.sum(axis=0) + iot_p.sum(axis=1)),
            (dtilde_iot.sum(axis=1) + self.ytilde_total_iot),
            rtol=1e-6,
        )  # errors are due to rounding and omission of household sector
        assert np.allclose(
            (
                dtilde_iot.sum(axis=0)
                + self.xtilde_iot.sum(axis=0)
                + self.o_iot.sum(axis=0)
            ),
            (dtilde_iot.sum(axis=1) + self.ytilde_total_iot),
            rtol=1e-6,
        )  # errors are due to rounding and omission of household sector
        assert np.allclose(
            self.gamma_d.sum(axis=0) + self.gamma_x.sum(axis=0), 1.0, atol=1e-9
        )
        assert (self.gamma_d >= 0).all().all()
        assert (self.gamma_x >= 0).all().all()
        # depends on p_tau
        cd_prod_fun = dtilde_iot.pow(self.gamma_d).prod(axis=0) * self.xtilde_iot.pow(
            self.gamma_x
        ).prod(axis=0)
        min_prod_fun = pd.concat(
            [
                dtilde_iot.multiply(1 / self.gamma_d).min(),
                self.xtilde_iot.multiply(1 / self.gamma_x).min(),
            ],
            axis=1,
        ).min(axis=1)
        sum_prod_fun = (
            dtilde_iot.multiply(self.gamma_d).sum()
            + self.xtilde_iot.multiply(self.gamma_x).sum()
        )
        lin_prod_fun = (
            1 - substitution_rate
        ) * min_prod_fun + substitution_rate * sum_prod_fun
        prod_fun = lin_prod_fun

        self.Lambda = (
            1
            / (1 - (self.o_iot.sum(axis=0) / self.q_iot) * p_tau)
            * (dtilde_iot.sum(axis=0) + self.xtilde_iot.sum(axis=0))
            / prod_fun
        )

        self.gamma_d_dict = {
            (i, j): self.gamma_d.loc[i, j] for i in Sector for j in Sector
        }
        self.gamma_x_dict = {(m, j): self.gamma_x.loc[m, j] for m in M for j in Sector}
        self.Lambda_dict = {i: self.Lambda[i] for i in Sector}

        weight_taxes = {
            i: p_tau * self.o_iot.loc[PrimaryInput.TAXES_PRODUCTION, i] / self.q_iot[i]
            for i in Sector
        }
        self.objective_per_sector = {
            i: self.indicator("xtilde", M.L, i)
            + self.indicator("xtilde", M.K, i)
            + self.indicator("q", i) * weight_taxes[i]
            for i in Sector
        }
        self.objective_c = -np.sum(list(self.objective_per_sector.values()), axis=0)
        assert self.objective_c.shape[0] == len(self.variables)
        self.max_gdp_per_sector = (
            self.xtilde_iot.loc[M.L]
            + self.xtilde_iot.loc[M.K]
            + self.o_iot.loc[PrimaryInput.TAXES_PRODUCTION]
        )
        self.max_gdp = self.max_gdp_per_sector.sum()

        self.c_production_function_lin(
            self.gamma_d_dict, self.gamma_x_dict, self.Lambda_dict, substitution_rate
        )
        self.c_input(self.o_iot, self.q_iot, p_tau)
        self.c_output(self.q_iot)
        self.c_demand(p_delta, self.ytilde_iot)

    def finalise_setup(
        self,
        p_lambda: pd.DataFrame,
        p_kappa: Mapping[Sector, float],
        wfh_productivity: Mapping[Sector, float],
    ):
        p_lambda_dict = {
            (i, s): p_lambda.loc[i, s] for i in Sector for s in LabourState
        }
        bounds = self.add_constraint(self.c_labour_compensation(p_lambda), self.bounds)
        bounds = self.add_constraint(
            self.c_labour_quantity(p_lambda_dict, wfh_productivity), bounds
        )
        bounds = self.add_constraint(self.c_capital(p_kappa=p_kappa), bounds)
        return self.objective_c, bounds.to_array(), self.lp_bounds


class PiecewiseLinearCobbDouglasGdpModel(BaseGdpModel):
    input_output_intermediate: pd.DataFrame
    input_output_primary: pd.DataFrame
    input_output_final: pd.DataFrame
    wfh: pd.DataFrame

    def __init__(
        self,
        p_delta: float = 1.0,
        p_tau: float = 1.0,
        substitution_rate: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p_delta = pd.DataFrame(p_delta, index=list(Sector), columns=list(FinalUse))
        self.p_tau = p_tau
        self.substitution_rate = substitution_rate
        self.setup = CobbDouglasLPSetup()
        self.results = IoGdpResult()
        self.labour_weight_region_age_per_sector_by_count: Mapping[
            Tuple[Sector, Region, Age]
        ] = {}
        self.labour_weight_region_age_per_sector_by_compensation: Mapping[
            Tuple[Sector, Region, Age]
        ] = {}

    def _get_datasources(self) -> Mapping[str, DataSource]:
        datasources = {
            "gdp": RegionSectorAgeDataSource,
            "workers": RegionSectorAgeDataSource,
            "growth_rates": SectorDataSource,
            "keyworker": SectorDataSource,
            "vulnerability": SectorDataSource,
            "wfh": SectorDataSource,
            "input_output_intermediate": DataFrameDataSource,
            "input_output_final": DataFrameDataSource,
            "input_output_primary": DataFrameDataSource,
        }
        return {k: v(k) for k, v in datasources.items()}

    def load(self, reader: Reader) -> None:
        super().load(reader)
        self.setup.initial_setup(
            self.input_output_primary,
            self.input_output_intermediate,
            self.input_output_final,
            self.p_delta,
            self.p_tau,
            self.substitution_rate,
        )
        # invariant: for a fixed sector, summing weights over all regions and ages gives 1
        self.labour_weight_region_age_per_sector_by_count = {
            (s, r, a): self.workers[r, s, a]
            / sum(
                self.workers[rr, s, aa] for (rr, aa) in itertools.product(Region, Age)
            )
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        self.labour_weight_region_age_per_sector_by_compensation = {
            (s, r, a): self.gdp[r, s, a]
            / sum(self.gdp[rr, s, aa] for (rr, aa) in itertools.product(Region, Age))
            for r, s, a in itertools.product(Region, Sector, Age)
        }

    def _postprocess_model_outputs(self, time, utilisations, r):
        x = pd.Series(r.x, index=self.setup.variables)
        # gdp
        max_gdp = self.setup.max_gdp
        gdp = {}
        for sector in Sector:
            gdp_for_sector = self.setup.objective_per_sector[sector].dot(r.x)
            # split outputs per region and age
            # note: these are absolute values to be interpreted relative to iot data loaded in setup
            for region in Region:
                for age in Age:
                    gdp[region, sector, age] = (
                        self.labour_weight_region_age_per_sector_by_compensation[
                            sector, region, age
                        ]
                        * gdp_for_sector
                    )

        # workers
        max_workers = sum(
            self.workers[key] for key in itertools.product(Region, Sector, Age)
        )
        workers = {
            (r, s, a): utilisations[LabourState.WORKING, r, s, a]
            * self.workers[r, s, a]
            for r, s, a in itertools.product(Region, Sector, Age)
        }

        # primary inputs
        # TODO: all these loops over Region and Age pairs are terribly slow, making postprocessing take longer than
        #       solving. these really need to be vectorized
        max_primary_inputs = {}
        for p, s in itertools.product(PrimaryInput, Sector):
            primary_input_value = self.setup.iot_p.loc[s, p]
            for r, a in itertools.product(Region, Age):
                max_primary_inputs[p, r, s, a] = (
                    primary_input_value
                    * self.labour_weight_region_age_per_sector_by_compensation[s, r, a]
                )
        primary_inputs = {}
        for s in Sector:
            # imports
            imports_sector = x[self.setup.V("x", M.I, s)]
            for r, a in itertools.product(Region, Age):
                primary_inputs[PrimaryInput.IMPORTS, r, s, a] = (
                    imports_sector
                    * self.labour_weight_region_age_per_sector_by_compensation[s, r, a]
                )
            # labour
            labour_sector = x[self.setup.V("xtilde", M.L, s)]
            for r, a in itertools.product(Region, Age):
                primary_inputs[PrimaryInput.COMPENSATION, r, s, a] = (
                    labour_sector
                    * self.labour_weight_region_age_per_sector_by_compensation[s, r, a]
                )
            # taxes
            o_fraction = self.p_tau * x[self.setup.V("q", s)] / self.setup.q_iot.loc[s]
            taxes_production = (
                o_fraction * self.setup.iot_p.loc[s, PrimaryInput.TAXES_PRODUCTION]
            )
            taxes_products = (
                o_fraction * self.setup.iot_p.loc[s, PrimaryInput.TAXES_PRODUCTS]
            )
            for r, a in itertools.product(Region, Age):
                primary_inputs[PrimaryInput.TAXES_PRODUCTION, r, s, a] = (
                    taxes_production
                    * self.labour_weight_region_age_per_sector_by_compensation[s, r, a]
                )
                primary_inputs[PrimaryInput.TAXES_PRODUCTS, r, s, a] = (
                    taxes_products
                    * self.labour_weight_region_age_per_sector_by_compensation[s, r, a]
                )
            # capital
            gross_operating_surplus = x[self.setup.V("xtilde", M.K, s)]
            if gross_operating_surplus >= 0.0:
                fixed_capital_consumption_share = self.setup.iot_p.loc[
                    s, PrimaryInput.FIXED_CAPITAL_CONSUMPTION
                ] / (
                    self.setup.iot_p.loc[s, PrimaryInput.FIXED_CAPITAL_CONSUMPTION]
                    + self.setup.iot_p.loc[s, PrimaryInput.NET_OPERATING_SURPLUS]
                )
                consumption_of_fixed_capital = (
                    gross_operating_surplus * fixed_capital_consumption_share
                )
                net_operating_surplus = gross_operating_surplus * (
                    1 - fixed_capital_consumption_share
                )
            else:
                consumption_of_fixed_capital = 0.0
                net_operating_surplus = gross_operating_surplus
            for r, a in itertools.product(Region, Age):
                primary_inputs[PrimaryInput.FIXED_CAPITAL_CONSUMPTION, r, s, a] = (
                    consumption_of_fixed_capital
                    * self.labour_weight_region_age_per_sector_by_compensation[s, r, a]
                )
                primary_inputs[PrimaryInput.NET_OPERATING_SURPLUS, r, s, a] = (
                    net_operating_surplus
                    * self.labour_weight_region_age_per_sector_by_compensation[s, r, a]
                )

        # final uses
        max_final_uses = {
            (u, s): self.setup.ytilde_iot.loc[s, u]
            for u, s in itertools.product(FinalUse, Sector)
        }
        final_uses = {}
        for s in Sector:
            total_final_use = x[self.setup.V("y", s)]
            for u in FinalUse:
                final_uses[u, s] = total_final_use * (
                    self.setup.ytilde_iot.loc[s, u] / self.setup.ytilde_total_iot.loc[s]
                )

        # compensation
        max_compensation = {
            (r, s, a): max_primary_inputs[PrimaryInput.COMPENSATION, r, s, a]
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        max_compensation_paid = max_compensation
        max_compensation_received = max_compensation
        max_compensation_subsidy = max_compensation
        compensation_paid = (
            {}
        )  # Note: this is redundant, also returned from primary inputs
        compensation_received = {}
        compensation_subsidy = {}
        for s in Sector:
            compensation = x[self.setup.V("xtilde", M.L, s)]
            for r, a in itertools.product(Region, Age):
                received_adjustment = (
                    utilisations[LabourState.WORKING, r, s, a]
                    + utilisations[LabourState.WFH, r, s, a]
                    + utilisations[LabourState.ILL, r, s, a]
                    + 0.8 * utilisations[LabourState.FURLOUGHED, r, s, a]
                ) / (
                    utilisations[LabourState.WORKING, r, s, a]
                    + utilisations[LabourState.WFH, r, s, a]
                    + utilisations[LabourState.ILL, r, s, a]
                )
                compensation_paid[r, s, a] = (
                    compensation
                    * self.labour_weight_region_age_per_sector_by_compensation[s, r, a]
                )
                compensation_received[r, s, a] = (
                    received_adjustment * compensation_paid[r, s, a]
                )
                compensation_subsidy[r, s, a] = (
                    compensation_received[r, s, a] - compensation_paid[r, s, a]
                )

        return IoGdpResult(
            gdp={time: gdp},
            workers={time: workers},
            growth_factor={},
            max_gdp=max_gdp,
            max_workers=max_workers,
            primary_inputs={time: primary_inputs},
            final_uses={time: final_uses},
            compensation_paid={time: compensation_paid},
            compensation_received={time: compensation_received},
            compensation_subsidy={time: compensation_subsidy},
            max_primary_inputs=max_primary_inputs,
            max_final_uses=max_final_uses,
            max_compensation_paid=max_compensation_paid,
            max_compensation_received=max_compensation_received,
            max_compensation_subsidy=max_compensation_subsidy,
        )

    def _simulate(
        self,
        time: int,
        utilisations: Mapping[Tuple[LabourState, Region, Sector, Age], float],
        capital: Mapping[Sector, float],
    ) -> IoGdpResult:

        # preprocess parameters
        p_lambda = pd.DataFrame(
            {
                l: {
                    s: np.sum(
                        [
                            self.labour_weight_region_age_per_sector_by_compensation[
                                s, region, age
                            ]
                            * utilisations[l, region, s, age]
                            for region in Region
                            for age in Age
                        ]
                    )
                    for s in Sector
                }
                for l in LabourState
            }
        )

        p_kappa = capital

        # setup linear program
        objective, bounds, lp_bounds = self.setup.finalise_setup(
            p_lambda, p_kappa, self.wfh
        )

        # run linear program
        r = linprog(
            c=objective,
            A_ub=bounds[0],
            b_ub=bounds[1],
            A_eq=bounds[2],
            b_eq=bounds[3],
            bounds=lp_bounds,
            method="interior-point",
            options={"maxiter": 1e4, "disp": False, "autoscale": False},
        )

        # check result
        if not r.success:
            raise ValueError(r.message)

        # postprocess model parameters
        return self._postprocess_model_outputs(time, utilisations, r)

    def simulate(
        self,
        time: int,
        lockdown: bool,
        utilisations: Mapping[Tuple[LabourState, Region, Sector, Age], float],
        **kwargs,
    ) -> None:
        try:
            capital = kwargs["capital"]
        except KeyError:
            if time == START_OF_TIME:
                capital = {s: 1.0 for s in Sector}
            else:
                raise ValueError("capital parameter required")

        result = self._simulate(time, utilisations, capital)
        # TODO: should this affect additional parameters to GDP?
        result.growth_factor[time], result.gdp[time] = self._apply_growth_factor(
            time, lockdown, result.gdp[time]
        )
        self.results.update(result)
