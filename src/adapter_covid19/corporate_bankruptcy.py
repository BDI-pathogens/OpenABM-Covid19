import abc
import copy
import logging
from typing import Optional, Mapping

import numpy as np
import pandas as pd
from scipy.stats import fisk, norm
from collections import Counter

from adapter_covid19.constants import DAYS_IN_A_YEAR, START_OF_TIME
from adapter_covid19.datasources import Reader, SectorDataSource
from adapter_covid19.enums import Sector, BusinessSize
from adapter_covid19.data_structures import SimulateState, CorporateState

LOGGER = logging.getLogger(__name__)


class BaseCorporateBankruptcyModel:
    def __init__(self, **kwargs):
        if kwargs:
            LOGGER.warning(f"Unused kwargs in {self.__class__.__name__}: {kwargs}")

    def load(self, reader: Reader) -> None:
        pass

    @abc.abstractmethod
    def simulate(self, state: SimulateState, **kwargs) -> None:
        """
        :param state:
        :return:
        """
        if kwargs:
            LOGGER.warning(f"Unused kwargs in {self.__class__.__name__}: {kwargs}")


class NaiveCorporateBankruptcyModel(BaseCorporateBankruptcyModel):
    def simulate(self, state: SimulateState, **kwargs) -> None:
        super().simulate(state, **kwargs)
        state.corporate_state = CorporateState(
            {s: 1 for s in Sector},
            {b: {s: 1 for s in Sector} for b in BusinessSize},
            {s: 1 for s in Sector},
        )


class CorporateBankruptcyModel(BaseCorporateBankruptcyModel):
    def __init__(
        self,
        beta: Optional[float] = None,
        large_cap_cash_surplus_months: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beta = beta or 1 + np.random.rand()
        # Numpy sinc is of pi*x, not of x
        self.sinc_theta = np.sinc(1 / self.beta)
        self.large_cap_cash_surplus_months = (
            large_cap_cash_surplus_months or 1 + np.random.randint(12)
        )
        self.sme_cash_buffer_days: Mapping[Sector, float] = {}
        self.large_cap_pct: Mapping[Sector, float] = {}
        self.employee_compensation: Mapping[Sector, float] = {}
        self.taxes_minus_subsidies: Mapping[Sector, float] = {}
        self.capital_consumption: Mapping[Sector, float] = {}
        self.value_added: Mapping[Sector, float] = {}
        self.lcap_clipped_cash_buffer: Mapping[Sector, float] = {}
        self.sme_clipped_cash_buffer: Mapping[Sector, float] = {}
        self.sme_count: Mapping[Sector, float] = {}
        self.largecap_count: Mapping[Sector, float] = {}
        self.outflows: Mapping[Sector, float] = {}
        self.turnover: pd.DataFrame = pd.DataFrame()
        self.sme_vulnerability: Mapping[Sector, float] = {}
        self.loan_guarantee_remaining: Mapping[Sector, float] = {}
        self.size_loan: Mapping[Sector, float] = {}
        self.sme_company_size: Mapping[Sector, float] = {}
        self.large_company_size: Mapping[Sector, float] = {}
        self.sme_company_received_loan: Mapping[Sector, float] = {}

    def load(self, reader: Reader) -> None:
        # primary inputs data
        io_df = reader.load_csv("input_output").set_index("Sector")
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
        value_added = (
            io_df.net_operating_surplus.apply(lambda x: max(x, 0))
            + io_df.employee_compensation
            + io_df.taxes_minus_subsidies
            + io_df.capital_consumption
        )
        self.value_added = {Sector[k]: v for k, v in value_added.to_dict().items()}

        # large cap cash buffer
        self.large_cap_pct = SectorDataSource("largecap_pct_turnover").load(reader)
        lcap_cash_buffer = (
            gross_operating_surplus
            * np.array([self.large_cap_pct[s] for s in Sector])
            * self.large_cap_cash_surplus_months
            / 12
        )
        lcap_clipped_cash_buffer = lcap_cash_buffer.apply(lambda x: max(x, 0))
        self.lcap_clipped_cash_buffer = {
            Sector[k]: v for k, v in lcap_clipped_cash_buffer.to_dict().items()
        }

        # sme cash buffer
        self.sme_cash_buffer_days = SectorDataSource("smallcap_cash").load(reader)
        sme_factor = np.array(
            [
                (self.sme_cash_buffer_days[s] / self.sinc_theta)
                / DAYS_IN_A_YEAR
                * (1 - self.large_cap_pct[s])
                for s in Sector
            ]
        )
        sme_cash_buffer = outflows * sme_factor
        sme_clipped_cash_buffer = sme_cash_buffer.apply(lambda x: max(x, 0))
        self.sme_clipped_cash_buffer = {
            Sector[k]: v for k, v in sme_clipped_cash_buffer.to_dict().items()
        }

        # company demographics
        self.turnover = reader.load_csv("company_size_and_turnover").set_index("Sector")
        self.sme_count = {
            Sector[k]: int(v)
            for k, v in self.turnover[self.turnover.min_size < 250]
            .groupby(["Sector"])["num_companies"]
            .sum()
            .to_dict()
            .items()
        }
        self.sme_count.update({s: 0 for s in set(Sector) - self.sme_count.keys()})
        self.largecap_count = {
            Sector[k]: int(v)
            for k, v in self.turnover[self.turnover.min_size >= 250]
            .groupby(["Sector"])["num_companies"]
            .sum()
            .to_dict()
            .items()
        }
        self.largecap_count.update(
            {s: 0 for s in set(Sector) - self.largecap_count.keys()}
        )
        sme_company_sector_counts = self.turnover[self.turnover.min_size < 250][
            ["min_size", "num_companies"]
        ]
        self.sme_company_size = {
            Sector[k]: v
            for k, v in np.repeat(
                sme_company_sector_counts.min_size,
                sme_company_sector_counts.num_companies,
            )
            .groupby(["Sector"])
            .apply(lambda x: np.array(x))
            .to_dict()
            .items()
        }
        large_company_sector_counts = self.turnover[self.turnover.min_size >= 250][
            ["min_size", "num_companies"]
        ]
        self.large_company_size = {
            Sector[k]: v
            for k, v in np.repeat(
                large_company_sector_counts.min_size,
                large_company_sector_counts.num_companies,
            )
            .groupby(["Sector"])
            .apply(lambda x: np.array(x))
            .to_dict()
            .items()
        }

        # simulate initial cash buffers
        self._init_sim()

        # vulnerabilities for new spending stimulus
        sme_vulnerability = reader.load_csv("sme_rate_payer_vulnerability").set_index(
            "Sector"
        )
        sme_vulnerability /= sme_vulnerability.sum()
        self.sme_vulnerability = {
            Sector[k]: v
            for k, v in sme_vulnerability["vulnerability"].to_dict().items()
        }

        # coronavirus business interruption loan scheme (CBILS) stimulus parameters
        self.loan_guarantee_remaining = 330e3
        self.size_loan = {
            0: 0.01,
            1: 0.05,
            2: 0.05,
            5: 0.05,
            10: 0.25,
            20: 0.25,
            50: 0.25,
            100: 0.25,
            200: 0.25,
        }
        self.sme_company_received_loan = {
            s: np.zeros(self.sme_count[s]) for s in Sector
        }

    def _init_sim(self) -> None:
        small_med = self._get_median_cash_buffer_days(False, self.outflows)
        large_med = self._get_median_cash_buffer_days(True, self.outflows)
        self.cash_state = {
            BusinessSize.large: {
                s: self._sim_cash_buffer(
                    self.largecap_count[s],
                    large_med[s],
                    self.lcap_clipped_cash_buffer[s],
                )
                for s in Sector
            },
            BusinessSize.sme: {
                s: self._sim_cash_buffer(
                    self.sme_count[s], small_med[s], self.sme_clipped_cash_buffer[s]
                )
                for s in Sector
            },
        }
        self.init_cash_state = copy.deepcopy(self.cash_state)
        large_cash_q5 = {
            s: np.quantile(self.init_cash_state[BusinessSize.large][s], 0.05)
            / DAYS_IN_A_YEAR
            for s in Sector
            if len(self.init_cash_state[BusinessSize.large][s])
        }
        sme_cash_q5 = {
            s: np.quantile(self.init_cash_state[BusinessSize.sme][s], 0.05)
            / DAYS_IN_A_YEAR
            for s in Sector
            if len(self.init_cash_state[BusinessSize.sme][s])
        }
        large_cash_iqr = {
            s: (
                np.quantile(self.init_cash_state[BusinessSize.large][s], 0.75)
                - np.quantile(self.init_cash_state[BusinessSize.large][s], 0.25)
            )
            / DAYS_IN_A_YEAR
            for s in Sector
            if len(self.init_cash_state[BusinessSize.large][s])
        }
        sme_cash_iqr = {
            s: (
                np.quantile(self.init_cash_state[BusinessSize.sme][s], 0.75)
                - np.quantile(self.init_cash_state[BusinessSize.sme][s], 0.25)
            )
            / DAYS_IN_A_YEAR
            for s in Sector
            if len(self.init_cash_state[BusinessSize.sme][s])
        }

        sme_init_cash_outgoing = {
            s: (1 - self.large_cap_pct[s])
            * {s: self.outflows[s] - self.value_added[s] for s in Sector}[s]
            / self.sme_count[s]
            / DAYS_IN_A_YEAR
            if self.sme_count[s]
            else 0
            for s in Sector
        }
        largecap_init_cash_outgoing = {
            s: self.large_cap_pct[s]
            * {s: self.outflows[s] - self.value_added[s] for s in Sector}[s]
            / self.largecap_count[s]
            / DAYS_IN_A_YEAR
            if self.largecap_count[s]
            else 0
            for s in Sector
        }

        self.cash_drag = {
            BusinessSize.large: {
                s: norm.rvs(
                    loc=(large_cash_q5[s] - largecap_init_cash_outgoing[s]),
                    scale=1e-6,
                    size=self.largecap_count[s],
                )
                if s in large_cash_q5 and large_cash_q5[s] == large_cash_q5[s]
                else np.zeros(self.largecap_count[s])
                for s in Sector
            },
            BusinessSize.sme: {
                s: norm.rvs(
                    loc=(sme_cash_q5[s] - sme_init_cash_outgoing[s]),
                    scale=1e-6,
                    size=self.sme_count[s],
                )
                if s in sme_cash_q5 and sme_cash_q5[s] == sme_cash_q5[s]
                else np.zeros(self.sme_count[s])
                for s in Sector
            },
        }

    # covid corporate financing facility
    def _apply_ccff(self) -> None:
        for s in Sector:
            sample = np.random.choice(
                np.where(self.cash_state[BusinessSize.large][s] > 0)[0],
                size=int(sum(self.cash_state[BusinessSize.large][s] > 0) * 0.2),
                replace=False,
            )
            self.cash_state[BusinessSize.large][s][sample] = np.inf
            self.init_cash_state[BusinessSize.large][s][sample] = np.inf

    def _sim_cash_buffer(
        self,
        size: int,
        median_solvency_days: float,
        cash_buffer: float,
        max_cash_buffer_days: Optional[float] = np.inf,
    ) -> np.array:
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
        self, lcap: bool, net_operating_surplus: Mapping[Sector, float] = None,
    ) -> Mapping[Sector, float]:
        mean_cash_buffer_days = self._get_mean_cash_buffer_days(
            lcap, net_operating_surplus
        )
        return {k: v * self.sinc_theta for k, v in mean_cash_buffer_days.items()}

    def _proportion_solvent(self, cash_buffer_sample: np.array) -> float:
        solvent = float(np.mean(cash_buffer_sample > 0))
        if np.isnan(solvent):
            return 1
        return solvent

    def _proportion_employees_job_exists(self) -> Mapping[Sector, float]:
        large_company_solvent = (
            pd.DataFrame(
                {
                    s: Counter(
                        self.large_company_size[s][
                            (self.cash_state[BusinessSize.large][s] > 0)
                        ]
                    )
                    for s in Sector
                    if s in self.large_company_size
                }
            )
            .T.stack()
            .reset_index()
        )
        sme_company_solvent = (
            pd.DataFrame(
                {
                    s: Counter(
                        self.sme_company_size[s][
                            (self.cash_state[BusinessSize.sme][s] > 0)
                        ]
                    )
                    for s in Sector
                    if s in self.sme_company_size
                }
            )
            .T.stack()
            .reset_index()
        )
        company_solvent = pd.concat([large_company_solvent, sme_company_solvent])
        company_solvent.columns = ["Sector", "min_size", "num_solvent_companies"]
        business_population = self.turnover.reset_index()
        business_population.Sector = business_population.Sector.apply(
            lambda x: Sector[x]
        )
        business_population_solvent = business_population.merge(
            company_solvent, on=["Sector", "min_size"], how="left"
        )
        business_population_solvent["num_employees_job_exists"] = (
            business_population_solvent.num_solvent_companies
            / business_population_solvent.num_companies
            * business_population_solvent.num_employees
        )
        sector_employees = business_population_solvent.groupby(["Sector"])[
            ["num_employees", "num_employees_job_exists"]
        ].sum()
        sector_employees["p_employees_job_exists"] = (
            sector_employees["num_employees_job_exists"]
            / sector_employees["num_employees"]
        )
        proportion_employees_job_exists = sector_employees[
            "p_employees_job_exists"
        ].to_dict()
        proportion_employees_job_exists.update(
            {s: 1.0 for s in set(Sector) - proportion_employees_job_exists.keys()}
        )
        return proportion_employees_job_exists

    def simulate(self, state: SimulateState, **kwargs,) -> None:
        super().simulate(state, **kwargs)
        try:
            # in the GDP model, net operating surplus is positive if corp is running profit
            # in the Corp model, net operting surplus is negative if corp is running profit
            net_operating_surplus = {
                k: -v for k, v in state.gdp_state.net_operating_surplus.items()
            }
        except AttributeError:
            raise ValueError(
                f"Incompatible model selection, {self.__class__.__name__}"
                + " requires a GDP model that implements `net_operating_surplus`"
            )
        if state.time == START_OF_TIME:
            naive_model = NaiveCorporateBankruptcyModel()
            naive_model.simulate(state, **kwargs)
            return
        # TODO: we should be able to deal with corp bankruptcies without lockdown
        if not state.lockdown:
            state.corporate_state = copy.deepcopy(state.previous.corporate_state)
            return
        if state.time == state.new_spending_day:
            self._new_spending_sector_allocation()
        if state.time == state.ccff_day:
            self._apply_ccff()
        if (state.time >= state.loan_guarantee_day) and self.loan_guarantee_remaining:
            self._loan_guarantees()
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

        state.corporate_state = CorporateState(
            capital_discount_factor=self._capital_discount_factor(proportion_solvent),
            proportion_solvent=proportion_solvent,
            proportion_employees_job_exists=self._proportion_employees_job_exists(),
        )

    def _capital_discount_factor(
        self, proportion_solvent: Mapping[BusinessSize, Mapping[Sector, float]],
    ) -> Mapping[Sector, float]:

        return {
            s: (
                proportion_solvent[BusinessSize.large][s] * self.large_cap_pct[s]
                + proportion_solvent[BusinessSize.sme][s] * (1 - self.large_cap_pct[s])
            )
            for s in Sector
        }

    def _update_state(self, net_operating_surplus: Mapping[Sector, float],) -> None:

        largecap_cash_outgoing = {
            s: self.large_cap_pct[s]
            * net_operating_surplus[s]
            / self.largecap_count[s]
            / DAYS_IN_A_YEAR
            if self.largecap_count[s]
            else 0
            for s in Sector
        }

        sme_cash_outgoing = {
            s: (1 - self.large_cap_pct[s])
            * net_operating_surplus[s]
            / self.sme_count[s]
            / DAYS_IN_A_YEAR
            if self.sme_count[s]
            else 0
            for s in Sector
        }

        for s in Sector:
            self.cash_state[BusinessSize.large][s] = np.maximum(
                np.minimum(
                    self.cash_state[BusinessSize.large][s]
                    - largecap_cash_outgoing[s]
                    - self.cash_drag[BusinessSize.large][s],
                    self.init_cash_state[BusinessSize.large][s],
                ),
                0,
            ) * (self.cash_state[BusinessSize.large][s] > 0)
            self.cash_state[BusinessSize.sme][s] = np.maximum(
                np.minimum(
                    self.cash_state[BusinessSize.sme][s]
                    - sme_cash_outgoing[s]
                    - self.cash_drag[BusinessSize.sme][s],
                    self.init_cash_state[BusinessSize.sme][s],
                ),
                0,
            ) * (self.cash_state[BusinessSize.sme][s] > 0)

    def _loan_guarantees(self):
        for s in Sector:
            valid_set = (1 - self.sme_company_received_loan[s]) * (
                self.cash_state[BusinessSize.sme][s] > 0
            )
            sample = np.random.choice(
                np.where(valid_set)[0],
                size=int(min(self.sme_count[s] * 0.01, sum(valid_set))),
                replace=False,
            )
            if not len(sample):
                return

            self.cash_state[BusinessSize.sme][s][sample] += np.array(
                [self.size_loan[i] for i in self.sme_company_size[s][sample]]
            )
            self.init_cash_state[BusinessSize.sme][s][sample] += np.array(
                [self.size_loan[i] for i in self.sme_company_size[s][sample]]
            )
            self.sme_company_received_loan[s][sample] += 1
            self.loan_guarantee_remaining -= np.sum(
                [self.size_loan[i] for i in self.sme_company_size[s][sample]]
            )

    def _new_spending_sector_allocation(self) -> None:
        stimulus_amounts = [0.01, 0.25]
        for s in Sector:
            try:
                sector_turnover = self.turnover.loc[Sector(s).name, :]
            except:
                continue
            if not self.sme_vulnerability[s]:
                continue
            turnover_weights = np.array(
                [
                    sector_turnover[(sector_turnover.min_size < 10)][
                        ["num_companies", "per_turnover"]
                    ].sum(),
                    sector_turnover[
                        (sector_turnover.min_size >= 10)
                        & (sector_turnover.min_size < 250)
                    ][["num_companies", "per_turnover"]].sum(),
                ]
            )

            weight_df = pd.DataFrame(turnover_weights)
            weight_df.columns = ["num_companies", "per_turnover"]
            weight_df["stimulus_amounts"] = stimulus_amounts

            weight_df["per_turnover"] /= weight_df["per_turnover"].sum()

            weight_df["max_stimulus"] = (
                weight_df["num_companies"] * weight_df["stimulus_amounts"]
            )

            weight_df["allocated_stimulus"] = (
                np.array([12e3, 20e3]) * self.sme_vulnerability[s]
            )

            n_solvent = sum(self.cash_state[BusinessSize.sme][s] > 0)

            breaks = list(
                np.floor(
                    (
                        (
                            weight_df[["max_stimulus", "allocated_stimulus"]].min(
                                axis=1
                            )
                            / weight_df["stimulus_amounts"]
                        )
                        / self.sme_count[s]
                        * n_solvent
                    )
                )
            )

            cash_stimulus = np.repeat(
                stimulus_amounts + [0], breaks + [n_solvent - sum(breaks)]
            )

            self.cash_state[BusinessSize.sme][s][
                self.cash_state[BusinessSize.sme][s] > 0
            ] += cash_stimulus
