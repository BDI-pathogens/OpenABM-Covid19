import itertools
import logging
from typing import Mapping, Tuple

from adapter_covid19.constants import START_OF_TIME
from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel
from adapter_covid19.data_structures import (
    SimulateState,
    Utilisations,
)
from adapter_covid19.datasources import Reader
from adapter_covid19.enums import Region, Sector, Age, LabourState
from adapter_covid19.gdp import BaseGdpModel
from adapter_covid19.personal_insolvency import PersonalBankruptcyModel

LOGGER = logging.getLogger(__name__)


class Economics:
    def __init__(
        self,
        gdp_model: BaseGdpModel,
        corporate_model: CorporateBankruptcyModel,
        personal_model: PersonalBankruptcyModel,
        **kwargs,
    ):
        """
        Economics simulator

        Parameters
        ----------
        gdp_model: Model to simulate GDP given reduced working numbers
        corporate_model: Model to simulate corporate bankruptcies
        personal_model: Model to simulate personal bankruptcies
        """
        if kwargs:
            LOGGER.warning(f"Unused kwargs in {self.__class__.__name__}: {kwargs}")
        self.gdp_model = gdp_model
        self.corporate_model = corporate_model
        self.personal_model = personal_model

    def load(self, reader: Reader) -> None:
        """
        Load data required for simulation

        Parameters
        ----------
        reader: helper class to load data
        """
        self.gdp_model.load(reader)
        self.corporate_model.load(reader)
        self.personal_model.load(reader)

    def simulate(self, simulate_state: SimulateState,) -> None:
        return self._simulate(
            simulate_state, simulate_state.time, simulate_state.utilisations,
        )

    def add_unemployment(
        self,
        utilisations: Mapping[Tuple[LabourState, Region, Sector, Age], float],
        unemployment_per_sector: Mapping[Sector, float],
    ) -> Mapping[Tuple[LabourState, Region, Sector, Age], float]:
        # TODO: remove
        unemployed = {
            (LabourState.UNEMPLOYED, r, s, a): sum(
                utilisations[l, r, s, a] * unemployment_per_sector[s]
                for l in [LabourState.WORKING, LabourState.WFH, LabourState.FURLOUGHED]
            )
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        working = {
            (LabourState.WORKING, r, s, a): utilisations[LabourState.WORKING, r, s, a]
            * (1 - unemployment_per_sector[s])
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        wfh = {
            (LabourState.WFH, r, s, a): utilisations[LabourState.WFH, r, s, a]
            * (1 - unemployment_per_sector[s])
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        furloughed = {
            (LabourState.FURLOUGHED, r, s, a): utilisations[
                LabourState.FURLOUGHED, r, s, a
            ]
            * (1 - unemployment_per_sector[s])
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        new_utilisations = {
            (LabourState.ILL, r, s, a): utilisations[LabourState.ILL, r, s, a]
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        new_utilisations.update(unemployed)
        new_utilisations.update(working)
        new_utilisations.update(wfh)
        new_utilisations.update(furloughed)
        return new_utilisations

    def _simulate(
        self, state: SimulateState, time: int, utilisations: Utilisations,
    ) -> None:
        """
        Simulate the economy

        Parameters
        ----------
        state: state
        time: from 0 to inf. Must be called sequentially
        utilisations:
            Mapping from region, sector and age group to a number
            between 0 and 1 describing the proportion of the
            workforce in work
        """
        # There shouldn't really be any logic in this method; this should solely
        # provide the plumbing for the other three models

        self.gdp_model.simulate(state)
        self.corporate_model.simulate(state)
        self.personal_model.simulate(state)
