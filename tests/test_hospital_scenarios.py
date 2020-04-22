#!/usr/bin/env python3
"""
Tests of the individual-based model, COVID19-IBM

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be 
run by calling 'pytest' from project folder.  

Created: April 2020
Author: Dylan Feldner-Busztin
"""

import subprocess, pytest, os, sys
import numpy as np, pandas as pd

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_HOSPITAL_FILE = TEST_DIR +"/data/new_hospital_baseline_parameters.csv"
PYTHON_C_DIR = TEST_DIR.replace("tests","") + "src/COVID19"
SCENARIO_HOSPITAL_FILE = TEST_DIR + "/data/scenario_hospital_baseline_parameters.csv"
SCENARIO_FILE = TEST_DIR + "/data/scenario_baseline_parameters.csv"

TEST_DATA_FILE = TEST_DIR + "/data/baseline_parameters.csv"
PARAM_LINE_NUMBER = 1
DATA_DIR_TEST = TEST_DIR + "/data"
TEST_HOUSEHOLD_FILE = TEST_DIR + "/data/baseline_household_demographics.csv"
TEST_OUTPUT_FILE = TEST_DIR +"/data/test_output.csv"
TEST_OUTPUT_FILE_HOSPITAL = TEST_DIR +"/data/test_hospital_output.csv"
TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP = TEST_DIR +"/data/time_step_hospital_output.csv"
TEST_INTERACTIONS_FILE = TEST_DIR +"/data/interactions_Run1.csv"
TEST_INDIVIDUAL_FILE = TEST_DIR +"/data/individual_file_Run1.csv"
TEST_HCW_FILE = TEST_DIR +"/data/ward_output.csv"
SRC_DIR = TEST_DIR.replace("tests","") + "src"
EXECUTABLE = SRC_DIR + "/covid19ibm.exe"

sys.path.append(PYTHON_C_DIR)
from parameters import ParameterSet


# print(params.list_params())
# print(params.get_param("sd_time_hospital_transition"))
# params.set_param("sd_time_hospital_transition", 4)

class TestClass(object):
    """
    Test class for checking
    """

    def test_zero_infectivity(self):
        """
        Set infections outside hospital to zero, total infections should equal hospital infections
        """

        # Adjust baseline parameter
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("infectious_rate", 0.0)
        params.set_param("mean_infectious_period", 0.0)
        params.set_param("sd_infectious_period", 0.0)
        params.write_params(SCENARIO_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("waiting_infectivity_modifier", 0.0)
        h_params.set_param("general_infectivity_modifier", 0.0)
        h_params.set_param("icu_infectivity_modifier", 0.0)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model using baseline parameters, pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "r")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)

        # Create a dataframe out of the terminal output
        numHeader = 10
        numFooter = 27
        df_output = pd.read_csv(TEST_OUTPUT_FILE, 
            comment = "#", 
            sep = ",", 
            skiprows=numHeader, 
            skipfooter=numFooter, 
            engine='python')

        output = df_output["total_infected"].iloc[-1]
        expected_output = int(params.get_param("n_seed_infection"))
        
        np.testing.assert_equal(output, expected_output)





















