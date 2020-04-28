import abc
import copy
import itertools
import logging
import random
from dataclasses import dataclass
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
    gdp: MutableMapping[int, Mapping[Tuple[Region, Sector, Age], float]]  # G
    workers: MutableMapping[int, Mapping[Tuple[Region, Sector, Age], float]]  # n
    max_gdp: float
    max_workers: float

    def fraction_gdp_by_sector(self, time: int) -> Mapping[Sector, float]:
        return {
            s: sum(
                self.gdp[time][r, s, a] / self.max_gdp
                for r, a in itertools.product(Region, Age)
            )
            for s in Sector
        }


class BaseGDPBackboneMixin(abc.ABC):
    beta: Mapping[Sector, float]

    @abc.abstractmethod
    def load(self, reader: Reader) -> None:
        pass

    def adjust_gdp(
        self, time: int, gdp: Mapping[Tuple[Region, Sector, Age], float],
    ) -> Mapping[Tuple[Region, Sector, Age], float]:
        return {
            (r, s, a): gdp[(r, s, a)]
            * (1 + self.beta.get(s, 0.0) * (time - START_OF_TIME) / DAYS_IN_A_YEAR)
            for (r, s, a) in gdp.keys()
        }


class LinearGDPBackboneMixin(BaseGDPBackboneMixin):
    def load(self, reader: Reader) -> None:
        super().load(reader)
        self.beta = SectorDataSource("growth_rates").load(reader)


class BaseGdpModel(abc.ABC):
    gdp: Mapping[Tuple[Region, Sector, Age], float]
    workers: Mapping[Tuple[Region, Sector, Age], float]
    keyworker: Mapping[Sector, float]

    def __init__(self, lockdown_recovery_time: int = 1, optimal_recovery: bool = False):
        self.lockdown_recovery_time = lockdown_recovery_time
        self.optimal_recovery = optimal_recovery
        self.recovery_order: Sequence[Tuple[Region, Sector, Age]] = []
        self.results = GdpResult({}, {}, 0, 0)
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
        super().load(reader)
        for k, v in self.datasources.items():
            self.__setattr__(k, v.load(reader))
        self._check_data()
        self.results.max_gdp = sum(
            self.gdp[key] for key in itertools.product(Region, Sector, Age)
        )
        self.results.max_workers = sum(
            self.workers[key] for key in itertools.product(Region, Sector, Age)
        )
        gdp_per_worker = {k: self.gdp[k] / self.workers[k] for k in self.gdp}
        # Sort in terms of greatest productivity to least productivity
        self.recovery_order = sorted(
            gdp_per_worker, key=lambda x: gdp_per_worker[x], reverse=True
        )
        if not self.optimal_recovery:
            random.shuffle(self.recovery_order)

    def _apply_lockdown(
        self,
        time: int,
        lockdown: bool,
        lockdown_exit_time: int,
        utilisations: MutableMapping[Tuple[LabourState, Region, Sector, Age], float],
    ) -> Mapping[Tuple[LabourState, Region, Sector, Age], float]:
        if not lockdown and not lockdown_exit_time:
            return utilisations
        # We assume all utilisations are given as WFH=0, and here we adjust for WFH
        utilisations = copy.deepcopy(utilisations)
        work_utilisations = {
            (r, s, a): utilisations[LabourState.WORKING, r, s, a]
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        base_utilisations = {
            (r, s, a): u * self.keyworker[s]
            for (r, s, a), u in work_utilisations.items()
        }
        if lockdown:
            work_utilisations = base_utilisations
            wfh_utilisations = {
                (r, s, a): utilisations[LabourState.WORKING, r, s, a]
                - base_utilisations[r, s, a]
                for r, s, a in itertools.product(Region, Sector, Age)
            }
        elif lockdown_exit_time:
            full_utilisations = copy.deepcopy(work_utilisations)
            full_workers = sum(
                self.workers[k] * work_utilisations[k]
                for k in itertools.product(Region, Sector, Age)
            )
            current_workers = key_workers = sum(
                self.workers[k] * base_utilisations[k]
                for k in itertools.product(Region, Sector, Age)
            )
            work_utilisations = copy.deepcopy(base_utilisations)
            recovery_factor = min(
                (time - lockdown_exit_time) / self.lockdown_recovery_time, 1
            )
            to_send_back = key_workers + (full_workers - key_workers) * recovery_factor
            for key in self.recovery_order:
                if current_workers >= to_send_back:
                    break
                spare_capacity = (
                    full_utilisations[key] - work_utilisations[key]
                ) * self.workers[key]
                increase = min(spare_capacity, to_send_back - current_workers)
                work_utilisations[key] += increase / self.workers[key]
                if work_utilisations[key] > 1 + 1e-6:
                    raise ValueError(
                        f"Utilisation > 1: {key}: {work_utilisations[key]}"
                    )
                current_workers += increase
            wfh_utilisations = {
                (r, s, a): utilisations[LabourState.WORKING, r, s, a]
                - work_utilisations[r, s, a]
                for r, s, a in itertools.product(Region, Sector, Age)
            }
        for r, s, a in itertools.product(Region, Sector, Age):
            utilisations[LabourState.WORKING, r, s, a] = work_utilisations[r, s, a]
            utilisations[LabourState.WFH, r, s, a] = wfh_utilisations[r, s, a]
            assert np.isclose(sum(utilisations[l, r, s, a] for l in LabourState), 1)
        return utilisations

    @abc.abstractmethod
    def simulate(
        self,
        time: int,
        lockdown: bool,
        lockdown_exit_time: int,
        utilisations: Mapping[Tuple[LabourState, Region, Sector, Age], float],
    ) -> None:
        # utilisations = self._apply_lockdown(time, lockdown, lockdown_exit_time, utilisations)
        pass


class LinearGdpModel(BaseGdpModel, LinearGDPBackboneMixin):
    gdp: Mapping[Tuple[Region, Sector, Age], float]
    workers: Mapping[Tuple[Region, Sector, Age], float]
    wfh: Mapping[str, float]
    vulnerability: Mapping[str, float]

    def __init__(self, lockdown_recovery_time: int = 1, optimal_recovery: bool = False):
        super().__init__(lockdown_recovery_time, optimal_recovery)

    def _get_datasources(self) -> Mapping[str, DataSource]:
        datasources = {
            "gdp": RegionSectorAgeDataSource,
            "workers": RegionSectorAgeDataSource,
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
        lockdown_exit_time: int,
        utilisations: Mapping[Tuple[LabourState, Region, Sector, Age], float],
    ) -> None:
        utilisations = self._apply_lockdown(
            time, lockdown, lockdown_exit_time, utilisations
        )
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
        if not lockdown:
            gdp = self.adjust_gdp(time, gdp)
        self.results.gdp[time] = gdp
        self.results.workers[time] = workers


class SupplyDemandGdpModel(BaseGdpModel, LinearGDPBackboneMixin):
    gdp: Mapping[Tuple[Region, Sector, Age], float]
    workers: Mapping[Tuple[Region, Sector, Age], float]
    wfh: Mapping[Sector, float]
    vulnerability: Mapping[Sector, float]
    supply: np.array
    demand: np.array

    def __init__(
        self,
        lockdown_recovery_time: int = 1,
        optimal_recovery: bool = False,
        theta: float = 1.2,
    ):
        super().__init__(lockdown_recovery_time, optimal_recovery)
        self.theta = theta

    def _get_datasources(self) -> Mapping[str, DataSource]:
        datasources = {
            "gdp": RegionSectorAgeDataSource,
            "workers": RegionSectorAgeDataSource,
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
        lockdown_exit_time: int,
        utilisations: Mapping[Tuple[LabourState, Region, Sector, Age], float],
    ) -> None:
        utilisations = self._apply_lockdown(
            time, lockdown, lockdown_exit_time, utilisations
        )
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
        if not lockdown:
            gdp = self.adjust_gdp(time, gdp)
        self.results.gdp[time] = gdp
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
        p_substitute: float,
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
                            p_substitute
                            * 1
                            / gamma_d_dict[jprime, i]
                            * self.indicator("d", jprime, i)
                            + (1 - p_substitute)
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
                            p_substitute
                            * 1
                            / gamma_x_dict[mprime, i]
                            * self.indicator("x", mprime, i)
                            + (1 - p_substitute)
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

    def c_capital(self, p_kappa: pd.Series):
        const = np.array([p_kappa.loc[i] * self.xtilde_iot.loc[M.K, i] for i in Sector])
        A = np.array([self.indicator("x", M.K, i) for i in Sector])
        normalization = np.array([1 / self.xtilde_iot.loc[M.K, i] for i in Sector])
        const = np.multiply(const, normalization)
        A = np.multiply(A, normalization[:, None])
        self.update_constraint(Bound(None, None, A, const))

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
        p_kappa: pd.Series,
        p_tau: float,
        p_substitute: float,
    ) -> None:
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
        ytilde_total_iot = ytilde_iot.sum(axis=1)
        gamma_d = dtilde_iot.div(dtilde_iot.sum(axis=0) + self.xtilde_iot.sum(axis=0))
        gamma_x = self.xtilde_iot.div(
            dtilde_iot.sum(axis=0) + self.xtilde_iot.sum(axis=0)
        )

        o_iot = iot_p[[PrimaryInput.TAXES_PRODUCTION, PrimaryInput.TAXES_PRODUCTS]].T
        q_iot = dtilde_iot.sum(axis=0) + iot_p.sum(axis=1)
        assert np.allclose(
            (dtilde_iot.sum(axis=0) + iot_p.sum(axis=1)),
            (dtilde_iot.sum(axis=1) + ytilde_total_iot),
            rtol=1e-6,
        )  # errors are due to rounding and omission of household sector
        assert np.allclose(
            (dtilde_iot.sum(axis=0) + self.xtilde_iot.sum(axis=0) + o_iot.sum(axis=0)),
            (dtilde_iot.sum(axis=1) + ytilde_total_iot),
            rtol=1e-6,
        )  # errors are due to rounding and omission of household sector
        assert np.allclose(gamma_d.sum(axis=0) + gamma_x.sum(axis=0), 1.0, atol=1e-9)
        assert (gamma_d >= 0).all().all()
        assert (gamma_x >= 0).all().all()
        # depends on p_tau
        cd_prod_fun = dtilde_iot.pow(gamma_d).prod(axis=0) * self.xtilde_iot.pow(
            gamma_x
        ).prod(axis=0)
        min_prod_fun = pd.concat(
            [
                dtilde_iot.multiply(1 / gamma_d).min(),
                self.xtilde_iot.multiply(1 / gamma_x).min(),
            ],
            axis=1,
        ).min(axis=1)
        sum_prod_fun = (
            dtilde_iot.multiply(gamma_d).sum() + self.xtilde_iot.multiply(gamma_x).sum()
        )
        lin_prod_fun = p_substitute * min_prod_fun + (1 - p_substitute) * sum_prod_fun
        prod_fun = lin_prod_fun

        Lambda = (
            1
            / (1 - (o_iot.sum(axis=0) / q_iot) * p_tau)
            * (dtilde_iot.sum(axis=0) + self.xtilde_iot.sum(axis=0))
            / prod_fun
        )

        gamma_d_dict = {(i, j): gamma_d.loc[i, j] for i in Sector for j in Sector}
        gamma_x_dict = {(m, j): gamma_x.loc[m, j] for m in M for j in Sector}
        Lambda_dict = {i: Lambda[i] for i in Sector}

        weight_taxes = {
            i: p_tau * o_iot.loc[PrimaryInput.TAXES_PRODUCTION, i] / q_iot[i]
            for i in Sector
        }
        self.objective_per_sector = {
            i: self.indicator("xtilde", M.L, i)
               + self.indicator("xtilde", M.K, i)
               + self.indicator("q", i) * weight_taxes[i]
            for i in Sector
        }
        self.objective_c = -np.sum(list(self.objective_per_sector.values()),axis=0)
        assert self.objective_c.shape[0] == len(self.variables)

        self.c_production_function_lin(
            gamma_d_dict, gamma_x_dict, Lambda_dict, p_substitute
        )
        self.c_input(o_iot, q_iot, p_tau)
        self.c_output(q_iot)
        self.c_capital(p_kappa)
        self.c_demand(p_delta, ytilde_iot)

    def finalise_setup(
        self, p_lambda: pd.DataFrame, wfh_productivity: Mapping[Sector, float]
    ):
        p_lambda_dict = {
            (i, s): p_lambda.loc[i, s] for i in Sector for s in LabourState
        }
        bounds = self.add_constraint(self.c_labour_compensation(p_lambda), self.bounds)
        bounds = self.add_constraint(
            self.c_labour_quantity(p_lambda_dict, wfh_productivity), bounds
        )
        bounds = bounds.to_array()
        return self.objective_c, bounds, self.lp_bounds


class CobbDouglasGdpModel(BaseGdpModel, LinearGDPBackboneMixin):
    input_output_intermediate: pd.DataFrame
    input_output_primary: pd.DataFrame
    input_output_final: pd.DataFrame

    def __init__(
        self,
        lockdown_recovery_time: int = 1,
        optimal_recovery: bool = False,
        p_kappa: float = 1.0,
        p_delta: float = 1.0,
        p_tau: float = 1.0,
        p_substitute: float = 0.5,
    ):
        super().__init__(lockdown_recovery_time, optimal_recovery)
        self.p_kappa = pd.Series(p_kappa, index=list(Sector))
        self.p_delta = pd.DataFrame(p_delta, index=list(Sector), columns=list(FinalUse))
        self.p_tau = p_tau
        self.p_substitute = p_substitute
        self.setup = CobbDouglasLPSetup()

    def _get_datasources(self) -> Mapping[str, DataSource]:
        datasources = {
            "gdp": RegionSectorAgeDataSource,
            "workers": RegionSectorAgeDataSource,
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
            self.p_kappa,
            self.p_tau,
            self.p_substitute,
        )

    def _simulate_workers(
        self, region: Region, sector: Sector, age: Age, utilisation: float
    ) -> float:
        return utilisation * self.workers[region, sector, age]

    def _simulate_gdp(
        self, utilisations: Mapping[Tuple[LabourState, Region, Sector, Age], float]
    ) -> Mapping[Tuple[Region, Sector, Age], float]:
        gdp = {}
        # TODO: get these weights from a data source
        # invariant: for a fixed sector, summing weights over all regions and ages gives 1
        weight_region_age_per_sector = {
            (sector,region,age): 1/(len(Region)*len(Age))
                for sector in Sector for region in Region for age in Age
        }
        p_lambda = pd.DataFrame(
            {
                l: {s: np.sum([weight_region_age_per_sector[s,region,age] * utilisations[l, region, s, age]
                               for region in Region for age in Age])
                    for s in Sector}
                for l in LabourState
            }
        )
        objective, bounds, lp_bounds = self.setup.finalise_setup(p_lambda, self.wfh)
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
        if not r.success:
            raise ValueError(r.message)
        for sector in Sector:
            gdp_for_sector = self.setup.objective_per_sector[sector].dot(
                r.x
            )
            for region in Region:
                for age in Age:
                    gdp[region, sector, age] = weight_region_age_per_sector[sector,region,age] * gdp_for_sector
        return gdp

    def simulate(
        self,
        time: int,
        lockdown: bool,
        lockdown_exit_time: int,
        utilisations: Mapping[Tuple[LabourState, Region, Sector, Age], float],
    ) -> None:
        utilisations = self._apply_lockdown(
            time, lockdown, lockdown_exit_time, utilisations
        )

        workers = {
            (r, s, a): self._simulate_workers(
                r, s, a, utilisations[LabourState.WORKING, r, s, a]
            )
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        gdp = self._simulate_gdp(utilisations)
        if not lockdown:
            gdp = self.adjust_gdp(time, gdp)
        self.results.gdp[time] = gdp
        self.results.workers[time] = workers
