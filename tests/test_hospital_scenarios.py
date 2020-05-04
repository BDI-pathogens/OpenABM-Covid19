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
TEST_HOSPITAL_FILE = TEST_DIR +"/data/hospital_baseline_parameters.csv"
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

# Use parameter file from Python C interface to adjust parameters
PYTHON_C_DIR = TEST_DIR.replace("tests","") + "src/COVID19"
sys.path.append(PYTHON_C_DIR)
from parameters import ParameterSet

# Files with adjusted parameters for each scenario
SCENARIO_FILE = TEST_DIR + "/data/scenario_baseline_parameters.csv"
SCENARIO_HOSPITAL_FILE = TEST_DIR + "/data/scenario_hospital_baseline_parameters.csv"


class TestClass(object):
    """
    Test class for checking
    """

    def test_zero_infectivity(self):
        """
        Set infections rate to zero, total infections should be equal to seed infections
        """

        # Adjust baseline parameter
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("infectious_rate", 0.0)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)

        # Construct the compilation command and compile
        # compile_command = "make clean; make all"
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command], 
            shell = True, 
            cwd = SRC_DIR, 
            capture_output = True
            )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {TEST_HOSPITAL_FILE}"

        # Call the model using baseline parameters, pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        # In the individual file, time_infected should be not equal to -1 in n_seed_infection number of cases
        df_individual_output = pd.read_csv(TEST_INDIVIDUAL_FILE)

        expected_output = int(params.get_param("n_seed_infection"))
        output = df_individual_output["time_infected"] != -1
        output = df_individual_output[output]
        output = len(output.index)

        np.testing.assert_equal(output, expected_output)


    def test_zero_beds(self):
        """
        Set hospital beds to zero
        """

        # Adjust baseline parameter
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_beds_covid_general_ward", 0)
        h_params.set_param("n_beds_covid_icu_ward", 0)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)

        # Construct the compilation command and compile
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command], 
            shell = True, 
            cwd = SRC_DIR, 
            capture_output = True
            )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        df_individual_output = pd.read_csv(TEST_INDIVIDUAL_FILE)

        n_patient_general = df_individual_output["time_general"] != -1
        n_patient_general = df_individual_output[n_patient_general]
        n_patient_general = len(n_patient_general.index)
        assert n_patient_general == 0

        n_patient_icu = df_individual_output["time_icu"] != -1
        n_patient_icu = df_individual_output[n_patient_icu]
        n_patient_icu = len(n_patient_icu.index)
        assert n_patient_icu == 0


    def test_zero_general_wards(self):
        """
        Set hospital general wards to zero
        """

        # Adjust baseline parameter
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_covid_general_wards", 0)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)

        # Construct the compilation command and compile
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command], 
            shell = True, 
            cwd = SRC_DIR, 
            capture_output = True
            )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        df_individual_output = pd.read_csv(TEST_INDIVIDUAL_FILE)

        n_patient_general = df_individual_output["time_general"] != -1
        n_patient_general = df_individual_output[n_patient_general]
        n_patient_general = len(n_patient_general.index)

        assert n_patient_general == 0


    def test_zero_icu_wards(self):
        """
        Set hospital icu wards to zero
        """

        # Adjust baseline parameter
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_covid_icu_wards", 0)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)

        # Construct the compilation command and compile
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command], 
            shell = True, 
            cwd = SRC_DIR, 
            capture_output = True
            )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        df_individual_output = pd.read_csv(TEST_INDIVIDUAL_FILE)

        n_patient_icu = df_individual_output["time_icu"] != -1
        n_patient_icu = df_individual_output[n_patient_icu]
        n_patient_icu = len(n_patient_icu.index)

        assert n_patient_icu == 0