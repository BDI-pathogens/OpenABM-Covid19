import itertools
from dataclasses import dataclass
from typing import Mapping, Tuple, MutableMapping

from adapter_covid19.constants import START_OF_TIME
from adapter_covid19.datasources import Reader
from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel
from adapter_covid19.gdp import BaseGdpModel, GdpResult
from adapter_covid19.personal_insolvency import (
    PersonalBankruptcyResults,
    PersonalBankruptcyModel,
)
from adapter_covid19.enums import Region, Sector, Age, LabourState, PrimaryInput


@dataclass
class EconomicsResult:
    gdp_result: GdpResult
    corporate_solvencies: MutableMapping[int, Mapping[Sector, float]]
    gdp: MutableMapping[int, Mapping[Tuple[Region, Sector, Age], float]]
    personal_bankruptcy: MutableMapping[int, Mapping[Region, PersonalBankruptcyResults]]

    def fraction_gdp_by_sector(self, time: int) -> Mapping[Sector, float]:
        return {
            s: sum(
                self.gdp[time][r, s, a] / self.gdp_result.max_gdp
                for r, a in itertools.product(Region, Age)
            )
            for s in Sector
        }


class Economics:
    def __init__(
        self,
        gdp_model: BaseGdpModel,
        corporate_model: CorporateBankruptcyModel,
        personal_model: PersonalBankruptcyModel,
    ):
        """
        Economics simulator

        Parameters
        ----------
        gdp_model: Model to simulate GDP given reduced working numbers
        corporate_model: Model to simulate corporate bankruptcies
        personal_model: Model to simulate personal bankruptcies
        """
        self.gdp_model = gdp_model
        self.corporate_model = corporate_model
        self.personal_model = personal_model
        self.results = EconomicsResult(
            gdp_model.results, {}, {}, personal_model.results
        )

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

    def simulate(
        self,
        time: int,
        lockdown: bool,
        utilisations: Mapping[Tuple[LabourState, Region, Sector, Age], float],
    ) -> None:
        """
        Simulate the economy

        Parameters
        ----------
        time: from 0 to inf. Must be called sequentially
        lockdown: is a lockdown in effect?
        utilisations:
            Mapping from region, sector and age group to a number
            between 0 and 1 describing the proportion of the
            workforce in work
        """
        self.gdp_model.simulate(
            time, lockdown, utilisations, capital=self.corporate_model.state.capital,
        )
        if time == START_OF_TIME:
            corporates_solvent_fraction = {s: 1 for s in Sector}
        else:
            primary_inputs = self.gdp_model.results.primary_inputs[time - 1]
            negative_net_operating_surplus = {
                s: -sum(
                    [
                        primary_inputs[PrimaryInput.NET_OPERATING_SURPLUS, r, s, a]
                        for r, a in itertools.product(Region, Age)
                    ]
                )
                for s in Sector
            }
            if lockdown:
                corporates_solvent_fraction = self.corporate_model.simulate(
                    negative_net_operating_surplus
                ).gdp_discount_factor
            else:
                corporates_solvent_fraction = self.results.corporate_solvencies[
                    time - 1
                ]
        self.results.corporate_solvencies[time] = corporates_solvent_fraction
        self.results.gdp[time] = {
            (r, s, a): self.gdp_model.results.gdp[time][r, s, a]
            * corporates_solvent_fraction[s]
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        if time == START_OF_TIME:
            self.personal_model.init_utilization(
                corporate_bankruptcy={s: 0 for s in Sector},
                utilization_ill=0,
                utilization_furloughed=0,
                utilization_wfh=0,
                utilization_working=1,
            )
            self.personal_model.simulate(time=time)
        else:
            corporate_bankruptcy = {
                s: 1 - self.results.corporate_solvencies[time - 1][s] for s in Sector
            }
            if lockdown:
                self.personal_model.init_utilization(
                    corporate_bankruptcy=corporate_bankruptcy,
                    utilization_ill=0.1,
                    utilization_furloughed=0.4,
                    utilization_wfh=0.4,
                    utilization_working=0.2,
                )
            else:
                self.personal_model.init_utilization(
                    corporate_bankruptcy=corporate_bankruptcy,
                    utilization_ill=0.1,
                    utilization_furloughed=0,
                    utilization_wfh=0.2,
                    utilization_working=0.8,
                )
            self.personal_model.simulate(time=time,)
