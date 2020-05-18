#!/usr/bin/env python3


"""
This example sets up a model from the base line parameters, sets a random rng seed
and then runs for 50 days, after 50 days lock down is turned on
"""


from COVID19.model import Parameters, Model, OccupationNetworkEnum
import json
from pathlib import Path
from random import randint
from tqdm import trange
from typing import Dict, Any
import os

import logging
import pandas as pd

LOGGER = logging.getLogger(__name__)

base_path = Path(__file__).parent.absolute()

BASELINE_PARAMS = base_path.parent / "tests/data/baseline_parameters.csv"
HOUSEHOLDS = base_path.parent / "tests/data/baseline_household_demographics.csv"
OUTPUT_DIR = base_path / "results"


def setup_params(updated_params: Dict[str, Any] = None):
    """[summary]
    set up a parameters object from the baseline file
    Customise any parameters supplied in the updated params dict 
    Keyword Arguments:
        updated_params {dict} -- [description] (default: None)
    Returns:
        Parameter set
    """
    p = Parameters(
        input_param_file=os.fspath(BASELINE_PARAMS),
        output_file_dir=os.fspath(OUTPUT_DIR),
        param_line_number=1,
        input_households=os.fspath(HOUSEHOLDS),
        read_param_file=True,
    )

    if updated_params:
        for k, v in updated_params.items():
            p.set_param(k, v)
    return p


def run_model(param_updates, n_steps=200, lockdown_at=None):
    params = setup_params(param_updates)
    # Create an instance of the Model
    model = Model(params)
    m_out = []
    for step in trange(n_steps):
        # Evaluate each step and save the results
        model.one_time_step()
        # LOGGER.info(
        #     model.one_time_step_results()
        # )  # If we want to see the results as we go uncomment this block 
        m_out.append(model.one_time_step_results())
        if lockdown_at:
            if step == lockdown_at:
                model.update_running_params("lockdown_on", 1)
                LOGGER.info(f"turning on lock down at step {step}")
                LOGGER.info(
                    f'lockdown_house_interaction_multiplier = {params.get_param("lockdown_house_interaction_multiplier")}'
                )
                LOGGER.info(
                    f'lockdown_random_network_multiplier = {params.get_param("lockdown_random_network_multiplier")}'
                )
                for oc_net in OccupationNetworkEnum:
                    LOGGER.info(
                        f'lockdown_occupation_multiplier{oc_net.name} = {params.get_param(f"lockdown_occupation_multiplier{oc_net.name}")}'
                    )
    df = pd.DataFrame(m_out)
    model.write_output_files()
    return df



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    param_updates = {"rng_seed": randint(0, 65000)}
    df = run_model(param_updates=param_updates, n_steps=200, lockdown_at=50)
    df.to_csv("results/covid_timeseries_Run1.csv", index=False)
    print(df)

