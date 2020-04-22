import collections
import itertools
import functools
import logging
import multiprocessing
import os
import pathlib
from typing import Mapping, Optional, Tuple

import click
import matplotlib.pyplot as plt
import pandas as pd
from COVID19 import simulation
from COVID19.model import Model, Parameters

from adapter_covid19.datasources import Reader
from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel
from adapter_covid19.economics import Economics
from adapter_covid19.gdp import LinearGdpModel, SupplyDemandGdpModel
from adapter_covid19.personal_insolvency import PersonalBankruptcyModel
from adapter_covid19.enums import Region, Sector, Age, Age10Y

logging.basicConfig(level=logging.INFO)

LOCKDOWN_PARMAETERS = {
    'quarantine_household_on_positive': 1,
    'quarantine_household_on_symptoms': 1,
    'self_quarantine_fraction': 0.8,
}

ECON_MODELS = {
    'linear': LinearGdpModel,
    'supplydemand': SupplyDemandGdpModel,
}

LOGGER = logging.getLogger('__name__')

def _write_csv(data: pd.DataFrame, filename: str, output_dir: str) -> None:
    data.to_csv(
        pathlib.Path(output_dir).joinpath(f'{filename}.csv'),
        index=False,
    )


def _output_econ_data(model: Economics, end_time: int, output_dir: str) -> None:
    output_data = {
        'gdp_by_sector': pd.DataFrame(
            [model.results.fraction_gdp_by_sector(i) for i in range(end_time)],
            index=range(end_time)
        ).T.sort_index().T.cumsum(axis=1),
        'corporate_solvencies_sector': pd.DataFrame(
            [model.results.corporate_solvencies[i] for i in model.results.corporate_solvencies]
        ),
        'personal_bankruptcies': pd.DataFrame([
            {r: model.results.personal_bankruptcy[i][r].personal_bankruptcy for r in Region}
            for i in model.results.personal_bankruptcy
        ]),
        'corporate_bankruptcies_region': pd.DataFrame([
            {r: model.results.personal_bankruptcy[i][r].corporate_bankruptcy for r in Region}
            for i in model.results.personal_bankruptcy
        ]),
    }
    for name, data in output_data.items():
        _write_csv(data, name, output_dir)


def plot_econ_data(results_dir: str):
    # GDP by sectors
    gdp_path = os.path.join(results_dir, 'gdp_by_sector.csv')
    gdp_data = pd.read_csv(gdp_path)
    fig, ax = plt.subplots()
    # Fill first sector
    ax.fill_between(gdp_data.index, gdp_data.iloc[:, 0] * 0, gdp_data.iloc[:, 0], label=gdp_data.columns[0])
    for i in range(1, gdp_data.shape[1]):
        ax.fill_between(gdp_data.index, gdp_data.iloc[:, i - 1], gdp_data.iloc[:, i], label=gdp_data.columns[i])
    ax.set_xlabel('Time / days')
    ax.set_ylabel('GDP / Pre-crisis GDP')
    ax.legend(ncol=2)

    # Affect of corporate bankruptcies on GDP, by sector
    fig, ax = plt.subplots()
    corp_bank_path = os.path.join(results_dir, 'corporate_solvencies_sector.csv')
    corp_bank_df = pd.read_csv(corp_bank_path)
    corp_bank_df.plot(ax=ax)

    # Fraction personal bankruptcies, by sector
    fig, ax = plt.subplots()
    personal_bank_path = os.path.join(results_dir, 'personal_bankruptcies.csv')
    personal_bank_df = pd.read_csv(personal_bank_path)
    personal_bank_df.plot(ax=ax)

    plt.show()


def _econ_worker(
        queues: Mapping[Region, multiprocessing.Queue],
        model: Economics,
        total_individuals: int,
        end_time: int,
        output_dir: str,
) -> None:
    utilisations = {}
    for step in range(end_time):
        lockdowns = {}
        for region, queue in queues.items():
            step_, results = queue.get()
            if step != step_:
                raise ValueError(f'Simulation offset: expected {step}, found {step_}, state: {results}')
            lockdowns[region] = results['lockdown']
            # total individuals is measured per region
            out_of_action = results['n_quarantine'] + results['n_hospital'] + results['n_critical'] + results['n_death']
            utilisations[region] = 1 - out_of_action / total_individuals

        if len(set(lockdowns.values())) != 1:
            raise NotImplementedError(
                f'Model cannot handle selective lockdown by region yet: step={step} lockdowns={lockdowns}'
            )
        lockdown = next(iter(lockdowns.values()))
        utilisations = {(r, s, a): utilisations[r] for r, s, a in itertools.product(Region, Sector, Age)}
        model.simulate(step, lockdown, utilisations)

    _output_econ_data(model, end_time, output_dir)


def _worker(
        args: Tuple[Region, Mapping[Age10Y, float], multiprocessing.Queue],
        output_dir: str,
        parameters_path: str,
        household_demographics_path: str,
        total_individuals: int,
        lockdown_start: Optional[int],
        lockdown_end: int,
        end_time: int,
) -> None:
    region, populations, queue = args
    # Prepare output paths for each region
    output_file_name = pathlib.Path(output_dir).joinpath(f'{region.name}_ts.csv')
    detailed_output_dir = pathlib.Path(output_dir).joinpath(f'{region.name}_detailed')
    os.makedirs(detailed_output_dir, exist_ok=True)

    # Init parameters from repo baseline
    params = Parameters(
        parameters_path,
        1,
        str(detailed_output_dir),
        household_demographics_path,
    )
    # Set more simulation params
    params.set_param('n_total', total_individuals)
    params.set_param('end_time', end_time)
    # FIXME: not writing detailed output
    params.set_param('sys_write_individual', 1)

    # Set populations
    for k, v in populations.items():
        params.set_param(k.value, v)

    # Activate lockdown if specified
    if lockdown_start is not None:
        for k, v in LOCKDOWN_PARMAETERS.items():
            params.set_param(k, v)
        params.set_param('lockdown_time_on', lockdown_start)
        params.set_param('lockdown_time_off', lockdown_end)

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
    outputs.drop(columns=['test_on_symptoms', 'app_turned_on'])
    outputs.to_csv(output_file_name, index=False)


@click.command()
@click.option(
    '--parameters',
    type=click.Path(
        exists=True, resolve_path=True, file_okay=True, dir_okay=False
    ),
    help='path to parameters file',
    default='../tests/data/baseline_parameters.csv',
    show_default=True,
)
@click.option(
    '--household-demographics',
    type=click.Path(
        exists=True, resolve_path=True, file_okay=True, dir_okay=False
    ),
    help='path to household demographics file',
    default='../tests/data/baseline_household_demographics.csv',
    show_default=True,
)
@click.option(
    '--outdir',
    type=click.Path(
        resolve_path=True, file_okay=False, dir_okay=True
    ),
    help='output results directory',
    default='./results',
    show_default=True,
)
@click.option(
    '--total-individuals',
    type=int,
    default=100_000,
    help='number of individuals in model',
    show_default=True,
)
@click.option(
    '--lockdown-start',
    type=int,
    default=None,
    help='time to start lockdown (None for no lockdown)',
    show_default=True,
)
@click.option(
    '--lockdown-end',
    type=int,
    default=None,
    help='time to end lockdown (ignored if --lockdown-start is None)',
    show_default=True,
)
@click.option(
    '--end-time',
    type=int,
    default=200,
    help='time to end simulation',
    show_default=True,
)
@click.option(
    '--econ-data-dir',
    type=click.Path(
        resolve_path=True, file_okay=False, dir_okay=True
    ),
    help='directory containing economics data',
    default='../src/adapter_covid19/data',
    show_default=True,
)
@click.option(
    '--gdp-model',
    type=click.Choice(ECON_MODELS.keys(), case_sensitive=False),
    default='linear',
    help='time to end simulation',
    show_default=True,
)
@click.option(
    '--n-workers',
    type=int,
    default=None,
    help='Number of workers (None means all CPUs)',
    show_default=True,
)
def main(
        parameters,
        household_demographics,
        outdir,
        total_individuals,
        lockdown_start,
        lockdown_end,
        end_time,
        econ_data_dir,
        gdp_model,
        n_workers,
) -> None:
    """
    Run simulations by region
    """

    # set up output directory and paths
    output_dir = click.format_filename(outdir)
    os.makedirs(output_dir, exist_ok=True)
    parameters_path = click.format_filename(parameters)
    household_demographics_path = click.format_filename(household_demographics)
    econ_data_dir = click.format_filename(econ_data_dir)

    # Read regional population distributions in age
    populations_df = pd.read_csv(os.path.join(econ_data_dir, 'populations.csv'))
    populations_by_region = {
        Region[k]: {Age10Y[kk]: vv for kk, vv in v.items()}
        for k, v in populations_df.set_index('region').T.to_dict().items()
    }
    queues = {r: multiprocessing.Queue() for r in Region}

    # Setup economics model
    reader = Reader(econ_data_dir)
    gdp_model = ECON_MODELS[gdp_model]()
    econ_model = Economics(gdp_model, CorporateBankruptcyModel(), PersonalBankruptcyModel())
    econ_model.load(reader)

    worker = functools.partial(
        _worker,
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

    spread_processes = [
        multiprocessing.Process(target=worker, args=((r, populations_by_region[r], queues[r]),))
        for r in Region
    ]
    econ_process = multiprocessing.Process(target=econ_worker, args=(queues,))
    processes = spread_processes + [econ_process]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    plot_econ_data(output_dir)


if __name__ == '__main__':
    main()
