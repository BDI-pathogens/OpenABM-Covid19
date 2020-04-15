#!/usr/bin/env python3


"""
This example sets up a model from the base line parameters, sets a random rng seed
and then runs for 50 days, after 50 days lock down is turned on
"""


from COVID19.model import Parameters, Model
import json
from pathlib import Path
import covid19
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
        input_household_file=os.fspath(HOUSEHOLDS),
        read_param_file=True,
    )

    if updated_params:
        for k, v in updated_params.items():
            p.set_param(k, v)
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    param_updates = {"rng_seed": randint(0, 65000)}
    params = setup_params(param_updates)
    # Create an instance of the Model
    model = Model(params)
    m_out = []
    # Run for 300 days
    for step in trange(300):
        # Evaluate each step and save the results
        model.one_time_step()
        # LOGGER.info(
        #     model.one_time_step_results()
        # )  # If we want to see the results as we go uncomment this block 
        m_out.append(model.one_time_step_results())
        if step == 50:
            model.update_running_params("lockdown_on", 1)
            LOGGER.info(f"turning on lock down at step {step}")
            LOGGER.info(
                f'lockdown_house_interaction_multiplier = {params.get_param("lockdown_house_interaction_multiplier")}'
            )
            LOGGER.info(
                f'lockdown_random_network_multiplier = {params.get_param("lockdown_random_network_multiplier")}'
            )
            LOGGER.info(
                f'lockdown_work_network_multiplier = {params.get_param("lockdown_work_network_multiplier")}'
            )
    df = pd.DataFrame(m_out)
    model.write_output_files()
    df.to_csv("results/timeseries_lockdown_at_50.csv", index=False)
    print(df)
