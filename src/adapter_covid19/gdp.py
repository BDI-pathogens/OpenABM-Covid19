from __future__ import annotations

import abc
import copy
import itertools
import logging
from dataclasses import dataclass
from typing import Tuple, Mapping, Sequence, Optional, Union, List

import numpy as np
import pandas as pd
from scipy.optimize import linprog, OptimizeResult

from adapter_covid19.constants import START_OF_TIME, DAYS_IN_A_YEAR
from adapter_covid19.data_structures import (
    SimulateState,
    GdpResult,
    IoGdpResult,
    IoGdpState,
    Utilisation,
)
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
    WorkerState,
    WorkerStateConditional,
)

LOGGER = logging.getLogger(__name__)


class BaseGdpModel(abc.ABC):
    gdp: Mapping[Tuple[Region, Sector, Age], float]
    workers: Mapping[Tuple[Region, Sector, Age], float]
    keyworker: Mapping[Sector, float]
    growth_rates: Mapping[Sector, float]

    def __init__(self, **kwargs):
        if kwargs:
            LOGGER.warning(f"Unused kwargs in {self.__class__.__name__}: {kwargs}")
        self.results = GdpResult()
        self.max_gdp = 0
        self.max_workers = 0
        self.datasources = self._get_datasources()
        for k, v in self.datasources.items():
            self.__setattr__(k, None)

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
        self.results.max_gdp = self.max_gdp = sum(
            self.gdp[key] for key in itertools.product(Region, Sector, Age)
        )
        self.results.max_workers = self.max_workers = sum(
            self.workers[key] for key in itertools.product(Region, Sector, Age)
        )

    def _apply_growth_factor(
        self, time: int, lockdown: bool, gdp: Mapping[Tuple[Region, Sector, Age], float]
    ) -> Tuple[Mapping[Sector, float], Mapping[Tuple[Region, Sector, Age], float]]:
        # TODO: remove, has been deprecated
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
    def simulate(self, state: SimulateState) -> None:
        pass


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
            + [
                ("lambda", l, i)
                for l in [LabourState.ILL, LabourState.WFH, LabourState.WORKING]
                for i in Sector
            ]
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
            "lambda": 0.0,
        }
        u_bounds = {
            "q": np.inf,
            "d": np.inf,
            "x": np.inf,
            "xtilde": np.inf,
            "y": np.inf,
            "lambda": 1.0,
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

    def c_demand(
        self, p_delta: Mapping[Tuple[Sector, FinalUse], float], ytilde_iot: pd.DataFrame
    ):
        const = np.array(
            [
                np.sum([p_delta[i, u] * ytilde_iot.loc[i, u] for u in FinalUse])
                for i in Sector
            ]
        )
        A = np.array([self.indicator("y", i) for i in Sector])
        normalization = np.array([1 / ytilde_iot.sum(axis=1).loc[i] for i in Sector])
        const = np.multiply(
            const, normalization
        )  # if normalized, should all be 1 if p_delta is 1
        A = np.multiply(A, normalization[:, None])
        return Bound(A, const, None, None)

    def c_labour_quantity(self, wfh_productivity: Mapping[Sector, float],) -> Bound:
        const = np.array([0.0 for i in Sector])
        A = np.array(
            [
                self.indicator("x", M.L, i)
                - (
                    self.indicator("lambda", LabourState.WORKING, i)
                    + wfh_productivity[i] * self.indicator("lambda", LabourState.WFH, i)
                )
                * self.xtilde_iot.loc[M.L, i]
                for i in Sector
            ]
        )
        normalization = np.array([1 / self.xtilde_iot.loc[M.L, i] for i in Sector])
        const = np.multiply(const, normalization)
        A = np.multiply(A, normalization[:, None])
        return Bound(None, None, A, const)

    def c_labour_compensation(self) -> Bound:
        const = np.array([0.0 for i in Sector])
        A = np.array(
            [
                self.indicator("xtilde", M.L, i)
                - (
                    self.indicator("lambda", LabourState.WORKING, i)
                    + self.indicator("lambda", LabourState.WFH, i)
                    + self.indicator("lambda", LabourState.ILL, i)
                )
                * self.xtilde_iot.loc[M.L, i]
                for i in Sector
            ]
        )
        normalization = np.array([1 / self.xtilde_iot.loc[M.L, i] for i in Sector])
        const = np.multiply(const, normalization)
        A = np.multiply(A, normalization[:, None])
        return Bound(None, None, A, const)

    def c_labour_constraints(self, p: Mapping[Sector, Utilisation]):
        p = {
            (w, i): v for i in Sector for w, v in p[i].to_dict().items()
        }  # Mapping[Tuple[WorkerStateConditional,Sector], float]
        # inequalities
        const_ub = []
        A_ub = []
        # HEALTHY_WFO == WORKING == (1-p_not_employed) * (1-p_wfh) * (1-p_ill_wfo) * (1-p_dead)
        # hence: WORKING <= (1-p_wfh) * (1-p_ill_wfo) * (1-p_dead)
        factor_wfo = [
            (1 - p[WorkerStateConditional.WFH, i])
            * (1 - p[WorkerStateConditional.ILL_WFO, i])
            * (1 - p[WorkerStateConditional.DEAD, i])
            for i in Sector
        ]
        const_ub += factor_wfo
        A_ub += [self.indicator("lambda", LabourState.WORKING, i) for i in Sector]
        # HEALTHY_WFH == WFH == (1-p_not_employed) * p_wfh * (1-p_ill_wfh) * (1-p_dead)
        # hence: WFH <= p_wfh * (1-p_ill_wfo) * (1-p_dead)
        factor_wfh = [
            p[WorkerStateConditional.WFH, i]
            * (1 - p[WorkerStateConditional.ILL_WFH, i])
            * (1 - p[WorkerStateConditional.DEAD, i])
            for i in Sector
        ]
        const_ub += factor_wfh
        A_ub += [self.indicator("lambda", LabourState.WFH, i) for i in Sector]
        # ILL_WFH + ILL_WFO == ILL == (1-p_not_employed) * (p_ill_wfo + p_ill_wfh) * (1-p_dead)
        # hence: ILL <= (p_ill_wfo + p_ill_wfh) * (1-p_dead)
        factor_ill = [
            (
                p[WorkerStateConditional.ILL_WFO, i]
                + p[WorkerStateConditional.ILL_WFH, i]
            )
            * (1 - p[WorkerStateConditional.DEAD, i])
            for i in Sector
        ]
        const_ub += factor_ill
        A_ub += [self.indicator("lambda", LabourState.ILL, i) for i in Sector]
        # equations
        const_eq = []
        A_eq = []
        # WFH and WFO are consistent
        for f_wfh, f_wfo, i in zip(factor_wfh, factor_wfo, Sector):
            const_eq.append(0.0)
            A_eq.append(
                self.indicator("lambda", LabourState.WFH, i) * f_wfo
                - self.indicator("lambda", LabourState.WORKING, i) * f_wfh
            )
        # WFH and ILL are consistent
        for f_wfh, f_ill, i in zip(factor_wfh, factor_ill, Sector):
            const_eq.append(0.0)
            A_eq.append(
                self.indicator("lambda", LabourState.WFH, i) * f_ill
                - self.indicator("lambda", LabourState.ILL, i) * f_wfh
            )
        # WFO and ILL are consistent
        for f_wfo, f_ill, i in zip(factor_wfo, factor_ill, Sector):
            const_eq.append(0.0)
            A_eq.append(
                self.indicator("lambda", LabourState.WORKING, i) * f_ill
                - self.indicator("lambda", LabourState.ILL, i) * f_wfo
            )
        # persist factors for use in post-processing
        self.labour_conditioning_factors = {}
        for f, s in zip(factor_wfo, Sector):
            self.labour_conditioning_factors[LabourState.WORKING, s] = f
        for f, s in zip(factor_wfh, Sector):
            self.labour_conditioning_factors[LabourState.WFH, s] = f
        for f, s in zip(factor_ill, Sector):
            self.labour_conditioning_factors[LabourState.ILL, s] = f
        # note: if all constraints are non-vacuous (no zero coefficients), one of these sets of constraints is redundant,
        # which leaves one degree of freedom per sector
        # ensure unemployment is at least as large as bound given exogenously
        # i.e. the following constraints hold
        for f_wfo, f_wfh, f_ill, i in zip(factor_wfo, factor_wfh, factor_ill, Sector):
            # the value of p_not_employed on the utilisation given as *input* is taken as a *lower bound* on
            # on the value of p_not_employed determined by the model as output
            not_employed_lower_bound = p[WorkerStateConditional.NOT_EMPLOYED, i]
            # self.indicator("lambda", LabourState.WORKING, i) <= (1 - not_employed_lower_bound) * f_wfo
            const_ub.append((1 - not_employed_lower_bound) * f_wfo)
            A_ub.append(self.indicator("lambda", LabourState.WORKING, i))
            # self.indicator("lambda", LabourState.WFH, i) <= (1 - not_employed_lower_bound) * f_wfh
            const_ub.append((1 - not_employed_lower_bound) * f_wfh)
            A_ub.append(self.indicator("lambda", LabourState.WFH, i))
            # self.indicator("lambda", LabourState.ILL, i) <= (1 - not_employed_lower_bound) * f_ill
            const_ub.append((1 - not_employed_lower_bound) * f_ill)
            A_ub.append(self.indicator("lambda", LabourState.ILL, i))
        # convert to numpy arrays
        const_ub = np.array(const_ub)
        A_ub = np.array(A_ub)
        const_eq = np.array(const_eq)
        A_eq = np.array(A_eq)
        return Bound(A_ub, const_ub, A_eq, const_eq)

    def initial_setup(
        self,
        iot_p: pd.DataFrame,
        dtilde_iot: pd.DataFrame,
        ytilde_iot: pd.DataFrame,
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
        self.gdp_per_sector = {
            i: self.indicator("xtilde", M.L, i)
            + self.indicator("xtilde", M.K, i)
            + self.indicator("q", i) * weight_taxes[i]
            for i in Sector
        }
        self.surplus_per_sector = {
            i: self.indicator("xtilde", M.K, i)
            for i in Sector  # households don't have capital input to production
        }
        self.objective_c = -np.sum(list(self.gdp_per_sector.values()), axis=0) - np.sum(list(self.surplus_per_sector.values()), axis=0)
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

    def finalise_setup(
        self,
        p_delta: Mapping[Tuple[Sector, FinalUse], float],
        p_kappa: Mapping[Sector, float],
        p_labour: Mapping[Sector, Utilisation],
        wfh_productivity: Mapping[Sector, float],
    ):
        bounds = self.add_constraint(self.c_labour_compensation(), self.bounds)
        bounds = self.add_constraint(self.c_labour_quantity(wfh_productivity), bounds)
        bounds = self.add_constraint(self.c_labour_constraints(p_labour), bounds)
        bounds = self.add_constraint(self.c_capital(p_kappa=p_kappa), bounds)
        bounds = self.add_constraint(
            self.c_demand(p_delta=p_delta, ytilde_iot=self.ytilde_iot), bounds
        )
        return self.objective_c, bounds.to_array(), self.lp_bounds

    def get_gdp(self, x):
        return pd.Series([self.gdp_per_sector[s].dot(x) for s in Sector], index=Sector)


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

    def _postprocess_model_outputs(
        self, state: SimulateState, res: OptimizeResult,
    ) -> IoGdpState:
        x = pd.Series(res.x, index=self.setup.variables)
        # gdp
        max_gdp = self.setup.max_gdp
        gdp = {}
        for sector in Sector:
            gdp_for_sector = self.setup.gdp_per_sector[sector].dot(res.x)
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
            (r, s, a): state.utilisations[r, s, a].to_lambdas()[WorkerState.HEALTHY_WFO]
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
        # TODO: review after introduction of labour variables
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
                utilisation = state.utilisations[r, s, a].to_lambdas()
                received_adjustment = (
                    utilisation[WorkerState.HEALTHY_WFO]
                    + utilisation[WorkerState.HEALTHY_WFH]
                    + utilisation[WorkerState.ILL_WFO]
                    + utilisation[WorkerState.ILL_WFH]
                    + 0.8 * utilisation[WorkerState.HEALTHY_FURLOUGHED]
                    + 0.8 * utilisation[WorkerState.HEALTHY_FURLOUGHED]
                ) / (
                    utilisation[WorkerState.HEALTHY_WFO]
                    + utilisation[WorkerState.HEALTHY_WFH]
                    + utilisation[WorkerState.ILL_WFH]
                    + utilisation[WorkerState.ILL_WFO]
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

        # utilisation and unemployment
        p_not_employed = {}
        for s in Sector:
            quotients = [
                (
                    x[self.setup.V("lambda", LabourState.WORKING, s)],
                    self.setup.labour_conditioning_factors[LabourState.WORKING, s],
                ),
                (
                    x[self.setup.V("lambda", LabourState.WFH, s)],
                    self.setup.labour_conditioning_factors[LabourState.WFH, s],
                ),
                (
                    x[self.setup.V("lambda", LabourState.ILL, s)],
                    self.setup.labour_conditioning_factors[LabourState.ILL, s],
                ),
            ]
            quotients = [q for q in quotients if q[1] > 0]
            quotients = sorted(quotients, key=lambda q: q[1], reverse=True)
            numerator, denominator = quotients[0]
            assert numerator >= 0.0
            assert denominator > 0.0
            assert (
                numerator / denominator <= 1.0 + 1e-2
            )  # tolerance due to rounding errors
            p_not_employed[s] = max(
                1.0 - numerator / denominator, 0.0
            )  # ensure strict non-negativity
            assert p_not_employed[s] >= 0.0
            # TODO: have corporate model account for p_furlough

        for r, s, a in itertools.product(Region, Sector, Age):
            not_employed_lower_bound = state.utilisations[r, s, a].p_not_employed
            if state.furlough:
                # if an intervention makes furloughing possible
                if p_not_employed[s] > 0.0:
                    p_furloughed = (
                        max(p_not_employed[s] - not_employed_lower_bound, 0.0)
                        / p_not_employed[s]
                    )
                else:
                    p_furloughed = 0.0
            else:
                p_furloughed = 0.0
            state.utilisations[r, s, a].p_not_employed = p_not_employed[s]
            state.utilisations[r, s, a].p_furloughed = p_furloughed
            assert 0 <= state.utilisations[r, s, a].p_not_employed <= 1
            assert 0 <= state.utilisations[r, s, a].p_furloughed <= 1

        # update gdp state
        gdp_state = IoGdpState(
            gdp=gdp,
            workers=workers,
            growth_factor={},
            max_gdp=max_gdp,
            max_workers=max_workers,
            primary_inputs=primary_inputs,
            final_uses=final_uses,
            compensation_paid=compensation_paid,
            compensation_received=compensation_received,
            compensation_subsidy=compensation_subsidy,
            max_primary_inputs=max_primary_inputs,
            max_final_uses=max_final_uses,
            max_compensation_paid=max_compensation_paid,
            max_compensation_received=max_compensation_received,
            max_compensation_subsidy=max_compensation_subsidy,
            _optimise_result=res,
        )

        state.gdp_state = gdp_state

    def _simulate(
        self,
        state: SimulateState,
        capital: Mapping[Sector, float],
        demand: Mapping[Tuple[Sector, FinalUse], float],
    ) -> IoGdpState:

        # preprocess parameters
        p_labour = {}
        for s in Sector:
            lambdas = state.utilisations[s]
            default_values = {
                WorkerStateConditional.WFH: self.keyworker[s],
                WorkerStateConditional.FURLOUGHED: 1.0,
                WorkerStateConditional.NOT_EMPLOYED: 0.0,
            }
            p_labour[s] = Utilisation.from_lambdas(lambdas, default_values)

        p_kappa = capital

        p_delta = demand

        # setup linear program
        objective, bounds, lp_bounds = self.setup.finalise_setup(
            p_delta, p_kappa, p_labour, self.wfh
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
        return self._postprocess_model_outputs(state, r)

    def simulate(self, state: SimulateState) -> None:
        # use capital parameter from corporate model
        if (
            state.previous is None
            or state.previous.corporate_state is None
            or state.previous.corporate_state.capital_discount_factor is None
        ):
            if state.time == START_OF_TIME:
                capital = {s: 1.0 for s in Sector}
            else:
                raise ValueError("capital parameter required")
        else:
            capital = copy.deepcopy(state.previous.corporate_state.capital_discount_factor)

        # apply assumed long-term growth rate to capital
        for s in Sector:
            capital[s] = capital[s] * (1 + self.growth_rates[s]) ** (
                (state.time - START_OF_TIME) / DAYS_IN_A_YEAR
            )

        # use unemployment from corporate model as *lower bound* on unemployment for gdp model
        if (
            state.previous is None
            or state.previous.corporate_state is None
            or state.previous.corporate_state.proportion_employees_job_exists is None
        ):
            # keep default value of p_not_employed as lower bound
            pass
        else:
            for r, s, a in itertools.product(Region, Sector, Age):
                state.utilisations[r, s, a].p_not_employed = min(
                    1.0,
                    max(
                        0.0,
                        1.0
                        - state.previous.corporate_state.proportion_employees_job_exists[
                            s
                        ],
                    ),
                )

        # use demand parameter from personal model
        if (
            state.previous is None
            or state.previous.personal_state is None
            or state.previous.personal_state.demand_reduction is None
        ):
            if state.time == START_OF_TIME:
                demand = {(i, u): 1.0 for i, u in itertools.product(Sector, FinalUse)}
            else:
                raise ValueError("demand parameter required")
        else:
            demand = {}
            for i in Sector:
                demand[i, FinalUse.E] = 1.0
                demand[i, FinalUse.K] = 1.0
                # TODO: apply output from personal model to household demand only
                demand[i, FinalUse.C] = 1.0 - np.nan_to_num(
                    state.previous.personal_state.demand_reduction[i]
                )

        # apply assumed long-term growth rate to demand
        for s, u in itertools.product(Sector, FinalUse):
            demand[s, u] = demand[s, u] * (1 + self.growth_rates[s]) ** (
                (state.time - START_OF_TIME) / DAYS_IN_A_YEAR
            )

        self._simulate(state, capital, demand)
