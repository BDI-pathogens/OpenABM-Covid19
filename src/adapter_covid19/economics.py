import itertools
from dataclasses import dataclass
from typing import Mapping, Tuple, MutableMapping

from adapter_covid19.constants import START_OF_TIME
from adapter_covid19.datasources import Reader
from adapter_covid19.corporate_bankruptcy import NaiveCorporateBankruptcyModel
from adapter_covid19.gdp import BaseGdpModel, GdpResult
from adapter_covid19.personal_insolvency import PersonalBankruptcyResults, PersonalBankruptcyModel
from adapter_covid19.enums import Region, Sector, Age


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
            ) for s in Sector
        }


class Economics:
    def __init__(
            self,
            gdp_model: BaseGdpModel,
            corporate_model: NaiveCorporateBankruptcyModel,
            personal_model: PersonalBankruptcyModel,
    ):
        self.first_lockdown_time = 10 ** 12
        self.lockdown_exited = False
        self.gdp_model = gdp_model
        self.corporate_model = corporate_model
        self.personal_model = personal_model
        self.results = EconomicsResult(gdp_model.results, {}, {}, personal_model.results)

    def load(self, reader: Reader):
        self.gdp_model.load(reader)
        self.corporate_model.load(reader)
        self.personal_model.load(reader)

    def _pre_simulation_checks(self, time: int, lockdown: bool) -> None:
        if time == START_OF_TIME and lockdown:
            raise ValueError('Economics model requires simulation to be started before lockdown')
        if self.lockdown_exited and lockdown:
            raise NotImplementedError('Bankruptcy/insolvency logic for toggling lockdown needs doing')
        if lockdown and time < self.first_lockdown_time:
            self.first_lockdown_time = time
        if self.first_lockdown_time < time and not lockdown:
            self.lockdown_exited = True

    def simulate(
            self,
            time: int,
            lockdown: bool,
            utilisations: Mapping[Tuple[Region, Sector, Age], float]
    ) -> None:
        self._pre_simulation_checks(time, lockdown)
        self.gdp_model.simulate(time, lockdown, utilisations)
        if time == START_OF_TIME:
            corporates_solvent_fraction = {s: 1 for s in Sector}
        else:
            gdp_discount = self.gdp_model.results.fraction_gdp_by_sector(time - 1)
            if lockdown:
                corporates_solvent_fraction = self.corporate_model.gdp_discount_factor(
                    time - self.first_lockdown_time, gdp_discount)
            else:
                corporates_solvent_fraction = self.results.corporate_solvencies[time - 1]
        self.results.corporate_solvencies[time] = corporates_solvent_fraction
        self.results.gdp[time] = {
            (r, s, a): self.gdp_model.results.gdp[time][r, s, a] * corporates_solvent_fraction[s]
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        if time == START_OF_TIME:
            self.personal_model.simulate(
                time=time, lockdown=lockdown,
                corporate_bankruptcy={s: 0 for s in Sector},
            )
        else:
            self.personal_model.simulate(
                time=time, lockdown=lockdown,
                corporate_bankruptcy={s: 1 - self.results.corporate_solvencies[time - 1][s] for s in Sector},
            )
