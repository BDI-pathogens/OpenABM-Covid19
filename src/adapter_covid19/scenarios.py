from __future__ import annotations

import itertools
from typing import Mapping, Tuple, Optional, MutableMapping

import numpy as np

from adapter_covid19.constants import START_OF_TIME
from adapter_covid19.data_structures import SimulateState, InitialiseState
from adapter_covid19.datasources import (
    SectorDataSource,
    Reader,
    RegionSectorAgeDataSource,
)
from adapter_covid19.enums import Sector, LabourState, Region, Age, EmploymentState


class Scenario:
    gdp: Mapping[Tuple[Region, Sector, Age], float]
    workers: Mapping[Tuple[Region, Sector, Age], float]
    furloughed: Mapping[Sector, float]
    keyworker: Mapping[Sector, float]

    def __init__(
        self,
        lockdown_recovery_time: int = 1,
        furlough_start_time: int = 1000,
        furlough_end_time: int = 1000,
        new_spending_day: int = 1000,
        ccff_day: int = 1000,
        loan_guarantee_day: int = 1000,
    ):
        self.datasources = {
            "gdp": RegionSectorAgeDataSource,
            "workers": RegionSectorAgeDataSource,
            "furloughed": SectorDataSource,
            "keyworker": SectorDataSource,
        }
        self.lockdown_recovery_time = lockdown_recovery_time
        self.lockdown_exited_time = 0
        self.furlough_start_time = furlough_start_time
        self.furlough_end_time = furlough_end_time
        self.new_spending_day = new_spending_day
        self.ccff_day = ccff_day
        self.loan_guarantee_day = loan_guarantee_day
        self._has_been_lockdown = False
        self.simulate_states: MutableMapping[int, SimulateState] = {}
        self._utilisations = {}  # For tracking / debugging
        for k in self.datasources:
            self.__setattr__(k, None)

    def load(self, reader: Reader) -> None:
        for k, v in self.datasources.items():
            self.__setattr__(k, v(k).load(reader))

    def _apply_lockdown(
        self,
        time: int,
        lockdown: bool,
        healthy: Mapping[Tuple[Region, Sector, Age], float],
        ill: Mapping[Tuple[Region, Sector, Age], float],
    ) -> Mapping[Tuple[LabourState, Region, Sector, Age], float]:
        # TODO: deprecated! remove in favour of Utilisation
        """
        Convert healthy/ill utilisations to working, ill, furloughed, wfh utilisations

        :param time:
        :param lockdown:
        :param healthy:
        :param ill:
        :return:
        """
        furlough_active = self.furlough_start_time <= time < self.furlough_end_time
        furloughed = self.furloughed if furlough_active else {s: 0.0 for s in Sector}

        utilisations = {
            k: 0 for k in itertools.product(LabourState, Region, Sector, Age)
        }
        if not lockdown and not self.lockdown_exited_time:
            # i.e. we're not in lockdown and we've not exited lockdown
            for r, s, a in itertools.product(Region, Sector, Age):
                utilisations[LabourState.WORKING, r, s, a] = healthy[r, s, a]
                utilisations[LabourState.ILL, r, s, a] = ill[r, s, a]
            return utilisations
        # We assume all utilisations are given as WFH=0, and here we adjust for WFH
        base_utilisations = {
            (r, s, a): u * self.keyworker[s] for (r, s, a), u in healthy.items()
        }
        # Lockdown utilisations
        work_utilisations = {
            (r, s, a): base_utilisations[r, s, a] * (1 - furloughed[s])
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        wfh_utilisations = {
            (r, s, a): (healthy[r, s, a] - base_utilisations[r, s, a])
            * (1 - furloughed[s])
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        furlough_utilisations = {
            (r, s, a): healthy[r, s, a] * furloughed[s]
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        if self.lockdown_exited_time:
            # Slowly bring people back to work and out of furlough
            # TODO: this assumes noone transitions from furloughed to unemployed
            factor = min(
                (time - self.lockdown_exited_time) / self.lockdown_recovery_time, 1
            )
            furlough_utilisations = {
                k: (1 - factor) * v for k, v in furlough_utilisations.items()
            }
            work_utilisations = {
                k: (base_utilisations[k] + factor * (healthy[k] - base_utilisations[k]))
                * (1 - furlough_utilisations[k])
                for k in itertools.product(Region, Sector, Age)
            }
            wfh_utilisations = {
                k: (healthy[k] - base_utilisations[k])
                * (1 - factor)
                * (1 - furlough_utilisations[k])
                for k in itertools.product(Region, Sector, Age)
            }

        for r, s, a in itertools.product(Region, Sector, Age):
            utilisations[LabourState.WORKING, r, s, a] = work_utilisations[r, s, a]
            utilisations[LabourState.WFH, r, s, a] = wfh_utilisations[r, s, a]
            utilisations[LabourState.FURLOUGHED, r, s, a] = furlough_utilisations[
                r, s, a
            ]
            utilisations[LabourState.ILL, r, s, a] = ill[r, s, a]
            assert np.isclose(sum(utilisations[l, r, s, a] for l in LabourState), 1)
        return utilisations

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

    def initialise(self) -> InitialiseState:
        # TODO: remove harcoded values
        return InitialiseState(
            personal_kwargs=dict(
                default_th=300,
                max_earning_furloughed=30_000,
                alpha=5,
                beta=20,
                min_expense_ratio=0.9,
            ),
            corporate_kwargs=dict(beta=1.4, large_cap_cash_surplus_months=6,),
        )

    def generate(
        self,
        time: int,
        dead: Mapping[Tuple[Region, Sector, Age], float],
        ill: Mapping[Tuple[Region, Sector, Age], float],
        lockdown: bool,
        furlough: bool,
        new_spending_day: int,
        ccff_day: int,
        loan_guarantee_day: int,
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
        )
        return simulate_state
