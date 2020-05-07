import logging

from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel
from adapter_covid19.data_structures import (
    SimulateState,
    Utilisations,
)
from adapter_covid19.datasources import Reader
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
