import copy
from typing import Optional, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy as sp

from adapter_covid19.constants import DAYS_IN_A_YEAR
from scipy.stats import fisk

from adapter_covid19.datasources import Reader, SectorDataSource
from adapter_covid19.enums import Sector, BusinessSize


@dataclass
class CorpInsolvencyState:
    gdp_discount_factor: Mapping[Sector, float]
    cash_buffer: Mapping[BusinessSize, Mapping[Sector, Sequence[float]]]
    proportion_solvent: Mapping[BusinessSize, Mapping[Sector, float]]
    capital: Mapping[Sector, float] = field(
        default_factory=lambda: {s: 1.0 for s in Sector}
    )


class NaiveCorporateBankruptcyModel:
    def __init__(self):
        self.state = CorpInsolvencyState(
            {s: 1 for s in Sector},
            {},
            {b: {s: 1 for s in Sector} for b in BusinessSize},
        )

    def load(self, reader: Reader) -> None:
        pass

    def simulate(
        self,
        days_since_lockdown: int,
        net_operating_surplus: Optional[Mapping[Sector, float]] = None,
    ) -> CorpInsolvencyState:
        """
        Amount to discount GDP by due to companies
        going insolvent on a specified number of days
        in the future

        :param days_since_lockdown:
        :param net_operating_surplus:
        :return:
        """
        return self.state


class CorporateBankruptcyModel:
    def __init__(
        self,
        beta: Optional[float] = None,
        large_cap_cash_surplus_months: Optional[float] = None,
    ):
        self.beta = beta or 1 + np.random.rand()
        super().__init__()
        theta = sp.pi / self.beta
        self.sinc_theta = np.sinc(theta)
        self.large_cap_cash_surplus_months = (
            large_cap_cash_surplus_months or 1 + np.random.randint(12)
        )
        self.small_cap_cash_buffer: Mapping[Sector, float] = {}
        self.large_cap_pct: Mapping[Sector, float] = {}
        self.employee_compensation: Mapping[Sector, float] = {}
        self.taxes_minus_subsidies: Mapping[Sector, float] = {}
        self.capital_consumption: Mapping[Sector, float] = {}
        self.value_added: Mapping[Sector, float] = {}
        self.lcap_clipped_cash_buffer: Mapping[Sector, float] = {}
        self.sme_clipped_cash_buffer: Mapping[Sector, float] = {}

    def load(self, reader: Reader) -> None:
        io_df = reader.load_csv("input_output").set_index("Sector")
        self.small_cap_cash_buffer = SectorDataSource("smallcap_cash").load(reader)
        self.large_cap_pct = SectorDataSource("largecap_pct_turnover").load(reader)
        self.sme_count = SectorDataSource("sme_count").load(reader)
        self.largecap_count = SectorDataSource("largecap_count").load(reader)
        self.employee_compensation = {
            Sector[k]: v for k, v in io_df.employee_compensation.to_dict().items()
        }
        self.taxes_minus_subsidies = {
            Sector[k]: v for k, v in io_df.taxes_minus_subsidies.to_dict().items()
        }
        self.capital_consumption = {
            Sector[k]: v for k, v in io_df.capital_consumption.to_dict().items()
        }
        outflows = (
            io_df.employee_compensation
            + io_df.taxes_minus_subsidies
            + io_df.capital_consumption
        )
        self.outflows = {Sector[k]: v for k, v in outflows.to_dict().items()}
        gross_operating_surplus = (
            io_df.net_operating_surplus.apply(lambda x: max(x, 0))
            + io_df.capital_consumption
        )
        lcap_cash_buffer = (
            gross_operating_surplus
            * np.array([self.large_cap_pct[s] for s in Sector])
            * self.large_cap_cash_surplus_months
            / 12
        )
        sme_factor = np.array(
            [
                (self.small_cap_cash_buffer[s] / self.sinc_theta)
                / DAYS_IN_A_YEAR
                * (1 - self.large_cap_pct[s])
                for s in Sector
            ]
        )
        sme_cash_buffer = outflows * sme_factor
        lcap_clipped_cash_buffer = lcap_cash_buffer.apply(lambda x: max(x, 0))
        sme_clipped_cash_buffer = sme_cash_buffer.apply(lambda x: max(x, 0))
        self.lcap_clipped_cash_buffer = {
            Sector[k]: v for k, v in lcap_clipped_cash_buffer.to_dict().items()
        }
        self.sme_clipped_cash_buffer = {
            Sector[k]: v for k, v in sme_clipped_cash_buffer.to_dict().items()
        }
        value_added = (
            io_df.net_operating_surplus.apply(lambda x: max(x, 0))
            + io_df.employee_compensation
            + io_df.taxes_minus_subsidies
            + io_df.capital_consumption
        )
        self.value_added = {Sector[k]: v for k, v in value_added.to_dict().items()}

        self._init_sim()

    def _init_sim(self) -> None:
        small_med = self._get_median_cash_buffer_days(False, self.outflows)
        large_med = self._get_median_cash_buffer_days(True, self.outflows)
        self.cash_state = {
            BusinessSize.large: {
                s: self._sim_cash_buffer(
                    100000, large_med[s], self.lcap_clipped_cash_buffer[s]
                )
                for s in Sector
            },
            BusinessSize.sme: {
                s: self._sim_cash_buffer(
                    100000, small_med[s], self.sme_clipped_cash_buffer[s]
                )
                for s in Sector
            },
        }
        self.init_cash_state = copy.deepcopy(self.cash_state)

    def _sim_cash_buffer(
        self,
        size: int,
        median_solvency_days: float,
        cash_buffer: float,
        max_cash_buffer_days: Optional[float] = np.inf,
    ) -> Sequence[float]:
        # Rejection sampling to get truncated log-logistic distribution of days till insolvency
        solvent_days = np.zeros((0,))
        while solvent_days.shape[0] < size:
            s = fisk.rvs(size=(size,), c=self.beta, scale=median_solvency_days)
            accepted = s[s <= max_cash_buffer_days]
            solvent_days = np.concatenate((solvent_days, accepted), axis=0)
        solvent_days = solvent_days[:size]

        total_solvent_days = sum(solvent_days)

        corp_cash_buffer = np.array(
            [days / total_solvent_days * cash_buffer for days in solvent_days]
        )
        return corp_cash_buffer

    def _get_mean_cash_buffer_days(
        self, lcap: bool, net_operating_surplus: Mapping[Sector, float],
    ) -> Mapping[Sector, float]:
        """

        :param lcap:
        :param net_operating_surplus:
        :return:
        """
        if lcap:
            size_modifier = self.large_cap_pct
            clipped_cash_buffer = self.lcap_clipped_cash_buffer
        else:
            size_modifier = {k: 1 - v for k, v in self.large_cap_pct.items()}
            clipped_cash_buffer = self.sme_clipped_cash_buffer
        return {
            # Added a nugget for when the denominator is 0
            s: DAYS_IN_A_YEAR
            * clipped_cash_buffer[s]
            / (size_modifier[s] * net_operating_surplus[s] - 1e-6)
            for s in Sector
        }

    def _get_median_cash_buffer_days(
        self,
        lcap: bool,
        net_operating_surplus: Optional[Mapping[Sector, float]] = None,
    ) -> Mapping[Sector, float]:
        mean_cash_buffer_days = self._get_mean_cash_buffer_days(
            lcap, net_operating_surplus
        )
        return {k: v * self.sinc_theta for k, v in mean_cash_buffer_days.items()}

    def _proportion_solvent(self, cash_buffer_sample: Sequence[float]) -> float:
        solvent = np.mean(cash_buffer_sample > 0)
        if np.isnan(solvent):
            return 1
        return solvent

    def simulate(
        self,
        days_since_lockdown: int,
        net_operating_surplus: Optional[Mapping[Sector, float]] = None,
    ) -> CorpInsolvencyState:
        """
        :param days_since_lockdown:
        :param net_operating_surplus:
        :return result:
        """
        self._update_state(net_operating_surplus)
        largecap_proportion_solvent = {
            s: self._proportion_solvent(self.cash_state[BusinessSize.large][s])
            for s in Sector
        }
        sme_proportion_solvent = {
            s: self._proportion_solvent(self.cash_state[BusinessSize.sme][s])
            for s in Sector
        }
        proportion_solvent = {
            BusinessSize.large: largecap_proportion_solvent,
            BusinessSize.sme: sme_proportion_solvent,
        }

        # TODO: add capital to state
        result = CorpInsolvencyState(
            self._gdp_discount_factor(proportion_solvent),
            self.cash_state,
            proportion_solvent,
        )

        self.state = result

        return result

    def _gdp_discount_factor(
        self, proportion_solvent: Mapping[BusinessSize, Mapping[Sector, float]],
    ) -> Mapping[Sector, float]:

        return {
            s: (
                proportion_solvent[BusinessSize.large][s] * self.large_cap_pct[s]
                + proportion_solvent[BusinessSize.sme][s] * (1 - self.large_cap_pct[s])
            )
            for s in Sector
        }

    def _update_state(
        self, net_operating_surplus: Optional[Mapping[Sector, float]] = None,
    ) -> None:

        largecap_cash_outgoing = {
            s: self.large_cap_pct[s] * net_operating_surplus[s] / 100000 / 365
            if self.largecap_count[s]
            else 0
            for s in Sector
        }

        sme_cash_outgoing = {
            s: (1 - self.large_cap_pct[s]) * net_operating_surplus[s] / 100000 / 365
            if self.sme_count[s]
            else 0
            for s in Sector
        }

        for s in Sector:
            self.cash_state[BusinessSize.large][s] = np.maximum(
                np.minimum(
                    self.cash_state[BusinessSize.large][s] - largecap_cash_outgoing[s],
                    self.init_cash_state[BusinessSize.large][s],
                ),
                0,
            ) * (self.cash_state[BusinessSize.large][s] > 0)
            self.cash_state[BusinessSize.sme][s] = np.maximum(
                np.minimum(
                    self.cash_state[BusinessSize.sme][s] - sme_cash_outgoing[s],
                    self.init_cash_state[BusinessSize.sme][s],
                ),
                0,
            ) * (self.cash_state[BusinessSize.sme][s] > 0)
