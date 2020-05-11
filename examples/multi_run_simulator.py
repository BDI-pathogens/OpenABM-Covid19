#!/usr/bin/env python3
"""Example script to evaluate models over time range with changes from the default parameter set
This example sets different random seeds for each evaulation and collects the data into pandas dataframes 
Note that individual traces, interactions etc are not saved due to the large number of them.
This is designed to be run to generate a stochastic window to gain a staticial interpretation of the model
Run the senario once with full output on to enable detailed knowledge of the model
"""
from COVID19.model import Parameters, Model
from tqdm import tqdm, trange
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import random
from pathlib import Path

base_path = Path(__file__).parent.absolute()

BASELINE_PARAMS = base_path.parent / "tests/data/baseline_parameters.csv"
HOUSEHOLDS = base_path.parent / "tests/data/baseline_household_demographics.csv"
OUTPUT_DIR = base_path / "results"


def setup_parameters(d: dict = None):
    # Set up Parameters
    # Override defaults that we pass in input dict
    p = Parameters(
        input_param_file=str(BASELINE_PARAMS),
        param_line_number=1,
        output_file_dir=str(OUTPUT_DIR),
        input_households=str(HOUSEHOLDS),
        read_param_file=True,
    )
    if d:
        for k, v in d.items():
            p.set_param(k, v)
    return p


def setup_model(d: dict = None):
    params = setup_parameters(d)
    params.set_param("sys_write_individual", 0)
    model = Model(params)
    return model


def run_model(d: dict = None):
    m = setup_model(d)
    results = []
    for _ in trange(100, desc="Model Progress"):
        m.one_time_step()
        results.append(m.one_time_step_results())
    return pd.DataFrame(results)


def run_many_inline(parameter_set_list, n_threads=None, progress_bar=True):
    if progress_bar:
        progress_monitor = tqdm
    else:
        progress_monitor = lambda x: x

    with ProcessPoolExecutor(n_threads) as ex:
        outputs = list(
            progress_monitor(
                ex.map(run_model, parameter_set_list),
                total=len(parameter_set_list),
                desc="Batch progress"
            )
        )
    return outputs


if __name__ == "__main__":

    print(BASELINE_PARAMS, HOUSEHOLDS)
    # Edit so we only run over 100k people, default is 1m but 10x speed increase for testing.
    # Remove n_total setting to run over larger population.
    params_list = [
        {"rng_seed": random.randint(0, 2 ** 32 - 1), "n_total": 100000}
        for _ in range(100)
    ]

    results_dataframes = run_many_inline(params_list, n_threads=4)

    # Ouput individual dataframes as CSVs
    for p, df in zip(params_list, results_dataframes):
        df.to_csv(OUTPUT_DIR / f"model_rng_seed_{p['rng_seed']}.csv", index=False)
