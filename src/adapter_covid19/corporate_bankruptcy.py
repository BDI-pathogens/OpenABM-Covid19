import copy
from typing import Optional, Mapping

import numpy as np
import scipy as sp
from scipy.stats import fisk

from adapter_covid19.datasources import Reader, SectorDataSource
from adapter_covid19.enums import Sector


class NaiveCorporateBankruptcyModel:
    def load(self, reader: Reader) -> None:
        pass

    def gdp_discount_factor(
            self,
            days_since_lockdown: int,
            gdp_discount: Optional[Mapping[Sector, float]] = None,
            comp_discount: Optional[Mapping[Sector, float]] = None,
            tax_discount: Optional[Mapping[Sector, float]] = None,
            consumption_cost_discount: Optional[Mapping[Sector, float]] = None,
    ) -> Mapping[Sector, float]:
        """
        Amount to discount GDP by due to companies
        going insolvent on a specified number of days
        in the future

        :param days_since_lockdown:
        :param gdp_discount:
        :param comp_discount:
        :param tax_discount:
        :param consumption_cost_discount:
        :return:
        """
        return {s: 1 for s in Sector}


class CorporateBankruptcyModel(NaiveCorporateBankruptcyModel):
    def __init__(
            self,
            beta: float = 1 + np.random.rand(),
            large_cap_cash_surplus_months: float = np.random.rand() * 12,
    ):
        self.beta = beta
        theta = sp.pi / self.beta
        self.sinc_theta = sp.sin(theta) / theta
        self.large_cap_cash_surplus_months = large_cap_cash_surplus_months
        self.small_cap_cash_buffer: Mapping[Sector, float] = {}
        self.large_cap_pct: Mapping[Sector, float] = {}
        self.employee_compensation: Mapping[Sector, float] = {}
        self.taxes_minus_subsidies: Mapping[Sector, float] = {}
        self.capital_consumption: Mapping[Sector, float] = {}
        self.value_added: Mapping[Sector, float] = {}
        self.lcap_clipped_cash_buffer: Mapping[Sector, float] = {}
        self.sme_clipped_cash_buffer: Mapping[Sector, float] = {}

    def load(self, reader: Reader) -> None:
        io_df = reader.load_csv('input_output').set_index('Sector')
        self.small_cap_cash_buffer = SectorDataSource('smallcap_cash').load(reader)
        self.large_cap_pct = SectorDataSource('largecap_pct_turnover').load(reader)
        self.employee_compensation = {Sector[k]: v for k, v in io_df.employee_compensation.to_dict().items()}
        self.taxes_minus_subsidies = {Sector[k]: v for k, v in io_df.taxes_minus_subsidies.to_dict().items()}
        self.capital_consumption = {Sector[k]: v for k, v in io_df.capital_consumption.to_dict().items()}
        outflows = io_df.employee_compensation + io_df.taxes_minus_subsidies + io_df.capital_consumption
        gross_operating_surplus = io_df.net_operating_surplus + io_df.capital_consumption
        lcap_cash_buffer = (
            gross_operating_surplus
            * np.array([self.large_cap_pct[s] for s in Sector])
            * self.large_cap_cash_surplus_months / 12
        )
        sme_factor = np.array([
            (self.small_cap_cash_buffer[s] / self.sinc_theta) / 365 * (1 - self.large_cap_pct[s]) for s in Sector
        ])
        sme_cash_buffer = outflows * sme_factor
        lcap_clipped_cash_buffer = lcap_cash_buffer.apply(lambda x: max(x, 0))
        sme_clipped_cash_buffer = sme_cash_buffer.apply(lambda x: max(x, 0))
        self.lcap_clipped_cash_buffer = {Sector[k]: v for k, v in lcap_clipped_cash_buffer.to_dict().items()}
        self.sme_clipped_cash_buffer = {Sector[k]: v for k, v in sme_clipped_cash_buffer.to_dict().items()}
        value_added = (
            io_df.net_operating_surplus
            + io_df.employee_compensation
            + io_df.taxes_minus_subsidies
            + io_df.capital_consumption
        )
        self.value_added = {Sector[k]: v for k, v in value_added.to_dict().items()}

    def _get_mean_cash_buffer_days(
            self,
            lcap: bool,
            gdp_discount: Optional[Mapping[Sector, float]],
            comp_discount: Optional[Mapping[Sector, float]],
            tax_discount: Optional[Mapping[Sector, float]],
            consumption_cost_discount: Optional[Mapping[Sector, float]],
    ) -> Mapping[Sector, float]:
        """

        :param lcap
        :param gdp_discount: 1.0 means no discount
        :param comp_discount:
        :param tax_discount:
        :param consumption_cost_discount:
        :return:
        """
        gdp_discount = gdp_discount or {s: 1 for s in Sector}
        comp_discount = comp_discount or {s: 1 for s in Sector}
        tax_discount = tax_discount or copy.deepcopy(gdp_discount)
        consumption_cost_discount = consumption_cost_discount or {s: 1 for s in Sector}
        if lcap:
            size_modifier = self.large_cap_pct
            clipped_cash_buffer = self.lcap_clipped_cash_buffer
        else:
            size_modifier = {k: 1 - v for k, v in self.large_cap_pct.items()}
            clipped_cash_buffer = self.sme_clipped_cash_buffer
        return {
            # Added a nugget for when the denominator is 0
            s: 365 * clipped_cash_buffer[s] / (
                size_modifier[s] * (
                    self.employee_compensation[s] * comp_discount[s]
                    + self.taxes_minus_subsidies[s] * tax_discount[s]
                    + self.capital_consumption[s] * consumption_cost_discount[s]
                    - self.value_added[s] * gdp_discount[s]
                ) - 1e-6)
            for s in Sector
        }

    def _get_median_cash_buffer_days(
            self,
            lcap: bool,
            gdp_discount: Optional[Mapping[Sector, float]],
            comp_discount: Optional[Mapping[Sector, float]],
            tax_discount: Optional[Mapping[Sector, float]],
            consumption_cost_discount: Optional[Mapping[Sector, float]],
    ) -> Mapping[Sector, float]:
        mean_cash_buffer_days = self._get_mean_cash_buffer_days(
            lcap, gdp_discount, comp_discount, tax_discount, consumption_cost_discount)
        return {k: v * self.sinc_theta for k, v in mean_cash_buffer_days.items()}

    def _proportion_solvent(self, days_since_lockdown: int, median_cash_buffer_day: float) -> float:
        solvent = fisk.sf(days_since_lockdown, self.beta, scale=median_cash_buffer_day)
        if np.isnan(solvent):
            return 0
        return solvent

    def gdp_discount_factor(
            self,
            days_since_lockdown: int,
            gdp_discount: Optional[Mapping[Sector, float]] = None,
            comp_discount: Optional[Mapping[Sector, float]] = None,
            tax_discount: Optional[Mapping[Sector, float]] = None,
            consumption_cost_discount: Optional[Mapping[Sector, float]] = None,
    ) -> Mapping[Sector, float]:
        """
        Proportion of Companies Insolvent on a Specified Number of Days in the Future
        :param days_since_lockdown:
        :param gdp_discount:
        :param comp_discount:
        :param tax_discount:
        :param consumption_cost_discount:
        :return:
        """
        if days_since_lockdown <= 0:
            return {s: 1 for s in Sector}
        lcap_median_cash_buffer_days = self._get_mean_cash_buffer_days(
            lcap=True,
            gdp_discount=gdp_discount,
            comp_discount=comp_discount,
            tax_discount=tax_discount,
            consumption_cost_discount=consumption_cost_discount
        )
        sme_median_cash_buffer_days = self._get_mean_cash_buffer_days(
            lcap=False,
            gdp_discount=gdp_discount,
            comp_discount=comp_discount,
            tax_discount=tax_discount,
            consumption_cost_discount=consumption_cost_discount
        )
        return {s: (
                self._proportion_solvent(days_since_lockdown, lcap_median_cash_buffer_days[s]) * self.large_cap_pct[s]
                + self._proportion_solvent(days_since_lockdown, sme_median_cash_buffer_days[s]) * (1 - self.large_cap_pct[s])
            ) for s in Sector}
