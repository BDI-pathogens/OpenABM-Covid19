import collections
import itertools
import functools
import logging
import multiprocessing
import os
import pathlib
from typing import Mapping, Optional, Tuple, Sequence, Union

import click
import matplotlib.pyplot as plt
import pandas as pd
from COVID19 import simulation
from COVID19.model import Model, Parameters

from adapter_covid19.datasources import Reader, RegionSectorAgeDataSource
from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel
from adapter_covid19.economics import Economics
from adapter_covid19.gdp import PiecewiseLinearCobbDouglasGdpModel
from adapter_covid19.personal_insolvency import PersonalBankruptcyModel
from adapter_covid19.enums import Region, Sector, Age, Age10Y, LabourState

logging.basicConfig(level=logging.INFO)

EXAMPLE_PARAMETERS = {
    "gdp": {},
    "corporate_bankruptcy": {"beta": 1.5, "large_cap_cash_surplus_months": 3},
    "personal_insolvency": {
        "saving": 10,
        "default_th": 100,
        "beta": 1,
        "gamma": 0.3,
        "utilization_ratio_ill": 0.1,
        "utilization_ratio_furloughed_lockdown": 0.4,
        "utilization_ratio_wfh_lockdown": 0.5,
        "utilization_ratio_working_lockdown": 0.1,
        "utilization_ratio_furloughed_no_lockdown": 0.0,
        "utilization_ratio_wfh_no_lockdown": 0.1,
        "utilization_ratio_working_no_lockdown": 0.9,
    },
}

LOGGER = logging.getLogger("__name__")


def _write_csv(data: pd.DataFrame, filename: str, output_dir: str) -> None:
    data.to_csv(
        pathlib.Path(output_dir).joinpath(f"{filename}.csv"), index=False,
    )


def _output_econ_data(model: Economics, end_time: int, output_dir: str) -> None:
    # TODO: The economic model generates a lot of data, and we only save a subset of it.
    # Either figure out how to save it nicely, or determine what's useful and save it
    output_data = {
        "gdp_by_sector": pd.DataFrame(
            [model.results.fraction_gdp_by_sector(i) for i in range(end_time)],
            index=range(end_time),
        )
        .T.sort_index()
        .T.cumsum(axis=1),
        "corporate_solvencies_by_sector": pd.DataFrame(
            [
                model.results.corporate_solvencies[i]
                for i in model.results.corporate_solvencies
            ]
        ),
        "personal_bankruptcies_by_region": pd.DataFrame(
            [
                {
                    r: model.results.personal_bankruptcy[i][r].personal_bankruptcy
                    for r in Region
                }
                for i in model.results.personal_bankruptcy
            ]
        ),
        "corporate_bankruptcies_by_region": pd.DataFrame(
            [
                {
                    r: model.results.personal_bankruptcy[i][r].corporate_bankruptcy
                    for r in Region
                }
                for i in model.results.personal_bankruptcy
            ]
        ),
    }
    for name, data in output_data.items():
        _write_csv(data, name, output_dir)


def plot_econ_data(
    model: Economics,
    total_individuals: int,
    end_time: int,
    data_dir: str,
    results_dir: str,
) -> Sequence[plt.Axes]:
    axes = []
    # Time series plot of overall values
    # Load all necessary data
    reader = Reader(data_dir)
    gdp_data = RegionSectorAgeDataSource("gdp").load(reader)
    gdp_per_sector = {
        s: sum(gdp_data[r, s, a] for r, a in itertools.product(Region, Age))
        for s in Sector
    }
    gdp_per_sector = {
        s: v / sum(gdp_per_sector.values()) for s, v in gdp_per_sector.items()
    }
    workers_data = RegionSectorAgeDataSource("gdp").load(reader)
    workers_per_region = {
        r: sum(workers_data[r, s, a] for s, a in itertools.product(Sector, Age))
        for r in Region
    }
    workers_per_region = {
        r: v / sum(workers_per_region.values()) for r, v in workers_per_region.items()
    }
    populations_path = os.path.join(data_dir, "populations.csv")
    people_per_region = {
        Region[k]: v
        for k, v in pd.read_csv(populations_path)
        .set_index("region")
        .sum(axis=1)
        .to_dict()
        .items()
    }
    people_per_region = {
        r: v / sum(people_per_region.values()) for r, v in people_per_region.items()
    }
    gdp = pd.Series(
        {
            k: sum(g.values()) / model.results.gdp_result.max_gdp
            for k, g in model.results.gdp.items()
        }
    )
    corporate_bankruptcies = 1 - pd.Series(
        {
            k: sum(cs[s] * gdp_per_sector[s] for s in Sector)
            for k, cs in model.results.corporate_solvencies.items()
        }
    )
    personal_bankruptcies = pd.Series(
        {
            k: sum(pb[r].personal_bankruptcy * workers_per_region[r] for r in Region)
            for k, pb in model.results.personal_bankruptcy.items()
        }
    )
    epidemic_data = {
        r: pd.read_csv(os.path.join(results_dir, f"{r.name}_ts.csv")) for r in Region
    }
    deaths = pd.Series(
        {
            i: sum(
                epidemic_data[r].loc[i, "n_death"]
                / total_individuals
                * people_per_region[r]
                for r in Region
            )
            for i in range(end_time)
        }
    )
    recoveries = pd.Series(
        {
            i: sum(
                epidemic_data[r].loc[i, "n_recovered"]
                / total_individuals
                * people_per_region[r]
                for r in Region
            )
            for i in range(end_time)
        }
    )
    # Write data
    _write_csv(gdp.to_frame(), "gdp", results_dir)
    _write_csv(corporate_bankruptcies.to_frame(), "corporate_bankruptcies", results_dir)
    _write_csv(personal_bankruptcies.to_frame(), "personal_bankruptcies", results_dir)
    _write_csv(deaths.to_frame(), "deaths", results_dir)
    _write_csv(recoveries.to_frame(), "recoveries", results_dir)
    # Plot
    fig, ax = plt.subplots()
    axes.append(ax)
    gdp.plot(ax=ax, label="gdp")
    corporate_bankruptcies.plot(ax=ax, label="corporate_bankruptcies")
    personal_bankruptcies.plot(ax=ax, label="personal_bankruptcies")
    (deaths * 100).plot(ax=ax, label="deaths * 100")
    recoveries.plot(ax=ax, label="recoveries")
    ax.set_xlabel("time / days")
    ax.legend()

    # GDP by sectors, area chart
    gdp_by_sector_path = os.path.join(results_dir, "gdp_by_sector.csv")
    gdp_by_sector = pd.read_csv(gdp_by_sector_path)
    fig, ax = plt.subplots()
    axes.append(ax)
    # First sector is special - need to fill between 0 and it
    ax.fill_between(
        gdp_by_sector.index,
        gdp_by_sector.iloc[:, 0] * 0,
        gdp_by_sector.iloc[:, 0],
        label=gdp_by_sector.columns[0],
    )
    for i in range(1, gdp_by_sector.shape[1]):
        ax.fill_between(
            gdp_by_sector.index,
            gdp_by_sector.iloc[:, i - 1],
            gdp_by_sector.iloc[:, i],
            label=gdp_by_sector.columns[i],
        )
    ax.set_xlabel("Time / days")
    ax.set_ylabel("GDP / Pre-crisis GDP")
    ax.legend(ncol=2)

    # Affect of corporate bankruptcies on GDP, by sector
    corp_bank_path = os.path.join(results_dir, "corporate_solvencies_by_sector.csv")
    corp_bank_df = pd.read_csv(corp_bank_path)
    fig, ax = plt.subplots()
    axes.append(ax)
    corp_bank_df.plot(ax=ax)

    # Fraction personal bankruptcies, by sector
    personal_bank_path = os.path.join(
        results_dir, "personal_bankruptcies_by_region.csv"
    )
    personal_bank_df = pd.read_csv(personal_bank_path)
    fig, ax = plt.subplots()
    axes.append(ax)
    personal_bank_df.plot(ax=ax)

    plt.show()
    return axes


def _econ_worker(
    queues: Mapping[Region, multiprocessing.Queue],
    model: Economics,
    total_individuals: int,
    end_time: int,
    output_dir: str,
) -> Economics:
    utilisations = {(l, r): 0 for l, r in itertools.product(LabourState, Region)}
    for step in range(end_time):
        lockdowns = {}
        for region, queue in queues.items():
            step_, results = queue.get()
            if step != step_:
                raise ValueError(
                    f"Simulation offset: expected {step}, found {step_}, state: {results}"
                )
            lockdowns[region] = results["lockdown"]
            # total individuals is measured per region
            ill = results["n_hospital"] + results["n_critical"] + results["n_death"]
            utilisations[LabourState.ILL, region] = ill / total_individuals
            utilisations[LabourState.WORKING] = 1 - ill / total_individuals

        if len(set(lockdowns.values())) != 1:
            raise NotImplementedError(
                f"Model cannot handle selective lockdown by region yet: step={step} lockdowns={lockdowns}"
            )
        lockdown = next(iter(lockdowns.values()))
        utilisations = {
            (r, s, a): utilisations[r]
            for r, s, a in itertools.product(Region, Sector, Age)
        }
        model.simulate(step, lockdown, utilisations)

    _output_econ_data(model, end_time, output_dir)
    return model


def _spread_worker(
    args: Tuple[Region, Mapping[Age10Y, float], multiprocessing.Queue],
    output_dir: str,
    parameters_path: str,
    household_demographics_path: str,
    total_individuals: int,
    lockdown_start: Optional[int],
    lockdown_end: Optional[int],
    end_time: int,
) -> None:
    region, populations, queue = args
    # Prepare output paths for each region
    output_file_name = pathlib.Path(output_dir).joinpath(f"{region.name}_ts.csv")
    detailed_output_dir = pathlib.Path(output_dir).joinpath(f"{region.name}_detailed")
    os.makedirs(detailed_output_dir, exist_ok=True)

    # Init parameters from repo baseline
    params = Parameters(
        parameters_path, 1, str(detailed_output_dir), household_demographics_path,
    )
    # Set more simulation params
    params.set_param("n_total", total_individuals)
    params.set_param("end_time", end_time)
    # FIXME: not writing detailed output
    params.set_param("sys_write_individual", 1)

    # Set populations
    for k, v in populations.items():
        params.set_param(k.value, v)

    # Activate lockdown if specified
    if lockdown_start is not None:
        params.set_param("lockdown_time_on", lockdown_start)
        if lockdown_end is None:
            lockdown_end = end_time + 1
        params.set_param("lockdown_time_off", lockdown_end)

    # Set up model
    model = simulation.COVID19IBM(model=Model(params))
    agent = simulation.Agent()
    results = collections.defaultdict(list)

    # Set up simulation
    current_state = model.start_simulation()
    current_action = agent.start_simulation(current_state)

    # Run simulation
    for step in range(end_time):
        current_state = model.step(current_action)
        current_action = agent.step(current_state)
        for key, value in current_state.items():
            results[key].append(value)
        queue.put((step, current_state))

    # Extract and write outputs
    outputs = pd.DataFrame(results)
    outputs.drop(columns=["test_on_symptoms", "app_turned_on"])
    outputs.to_csv(output_file_name, index=False)


@click.command()
@click.option(
    "--parameters",
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="path to parameters file",
    default="../tests/data/baseline_parameters.csv",
    show_default=True,
)
@click.option(
    "--household-demographics",
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    help="path to household demographics file",
    default="../tests/data/baseline_household_demographics.csv",
    show_default=True,
)
@click.option(
    "--outdir",
    type=click.Path(resolve_path=True, file_okay=False, dir_okay=True),
    help="output results directory",
    default="./results",
    show_default=True,
)
@click.option(
    "--total-individuals",
    type=int,
    default=100_000,
    help="number of individuals in model",
    show_default=True,
)
@click.option(
    "--lockdown-start",
    type=int,
    default=None,
    help="time to start lockdown (None for no lockdown)",
    show_default=True,
)
@click.option(
    "--lockdown-end",
    type=int,
    default=None,
    help="time to end lockdown (ignored if --lockdown-start is None)",
    show_default=True,
)
@click.option(
    "--end-time",
    type=int,
    default=200,
    help="time to end simulation",
    show_default=True,
)
@click.option(
    "--econ-data-dir",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    help="directory containing economics data",
    default="../tests/adapter_covid19/data",
    show_default=True,
)
@click.option(
    "--n-workers",
    type=int,
    default=None,
    help="Number of cpu cores to use (default None means all)",
    show_default=True,
)
def _main(
    parameters,
    household_demographics,
    outdir,
    total_individuals,
    lockdown_start,
    lockdown_end,
    end_time,
    econ_data_dir,
    n_workers,
) -> Sequence[plt.Axes]:
    """
    Run simulations by region
    """
    return main(
        parameters,
        household_demographics,
        outdir,
        total_individuals,
        lockdown_start,
        lockdown_end,
        end_time,
        econ_data_dir,
        n_workers,
    )


def main(
    parameters: str = "../tests/data/baseline_parameters.csv",
    household_demographics: str = "../tests/data/baseline_household_demographics.csv",
    outdir: str = "./results",
    total_individuals: int = 100_000,
    lockdown_start: Optional[int] = None,
    lockdown_end: Optional[int] = None,
    end_time: int = 200,
    econ_data_dir: str = "../tests/data/adapter_covid19",
    n_workers: Optional[int] = None,
    example_parameters: Optional[
        Mapping[str, Mapping[str, Union[float, int, bool, str]]]
    ] = None,
) -> Sequence[plt.Axes]:
    """
    Run simulations by region
    """

    # set up output directory and paths
    output_dir = click.format_filename(outdir)
    os.makedirs(output_dir, exist_ok=True)
    parameters_path = click.format_filename(parameters)
    household_demographics_path = click.format_filename(household_demographics)
    econ_data_dir = click.format_filename(econ_data_dir)
    if example_parameters is None:
        example_parameters = EXAMPLE_PARAMETERS

    # Read regional population distributions in age
    populations_df = pd.read_csv(os.path.join(econ_data_dir, "populations.csv"))
    populations_by_region = {
        Region[k]: {Age10Y[kk]: vv for kk, vv in v.items()}
        for k, v in populations_df.set_index("region").T.to_dict().items()
    }
    manager = multiprocessing.Manager()
    queues = {r: manager.Queue() for r in Region}

    # Setup economics model
    reader = Reader(econ_data_dir)
    gdp_model = PiecewiseLinearCobbDouglasGdpModel(**example_parameters["gdp"])
    cb_model = CorporateBankruptcyModel(**example_parameters["corporate_bankruptcy"])
    pb_model = PersonalBankruptcyModel(**example_parameters["personal_insolvency"])
    econ_model = Economics(gdp_model, cb_model, pb_model)
    econ_model.load(reader)

    spread_worker = functools.partial(
        _spread_worker,
        output_dir=output_dir,
        parameters_path=parameters_path,
        household_demographics_path=household_demographics_path,
        total_individuals=total_individuals,
        lockdown_start=lockdown_start,
        lockdown_end=lockdown_end,
        end_time=end_time,
    )

    econ_worker = functools.partial(
        _econ_worker,
        model=econ_model,
        total_individuals=total_individuals,
        end_time=end_time,
        output_dir=output_dir,
    )

    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.map(
            spread_worker, [(r, populations_by_region[r], queues[r]) for r in Region]
        )

    econ_worker(queues)

    return plot_econ_data(
        econ_model, total_individuals, end_time, econ_data_dir, output_dir
    )


if __name__ == "__main__":
    _main()
