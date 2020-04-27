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
TEST_DATA_FILE = TEST_DIR + "/data/baseline_parameters.csv"
PARAM_LINE_NUMBER = 1
DATA_DIR_TEST = TEST_DIR + "/data"
TEST_HOUSEHOLD_FILE = TEST_DIR + "/data/baseline_household_demographics.csv"
TEST_HOSPITAL_FILE = TEST_DIR +"/data/hospital_baseline_parameters.csv"
TEST_OUTPUT_FILE = TEST_DIR +"/data/test_output.csv"
TEST_OUTPUT_FILE_HOSPITAL = TEST_DIR +"/data/test_hospital_output.csv"
TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP = TEST_DIR +"/data/time_step_hospital_output.csv"
TEST_INTERACTIONS_FILE = TEST_DIR +"/data/interactions_Run1.csv"
TEST_INDIVIDUAL_FILE = TEST_DIR +"/data/individual_file_Run1.csv"
TEST_HCW_FILE = TEST_DIR +"/data/ward_output.csv"
SRC_DIR = TEST_DIR.replace("tests","") + "src"
EXECUTABLE = SRC_DIR + "/covid19ibm.exe"

# Construct the compilation command and compile
compile_command = "make clean; make all HOSPITAL_ON=1"
completed_compilation = subprocess.run([compile_command], 
    shell = True, 
    cwd = SRC_DIR, 
    capture_output = True
    )

# Construct the executable command
EXE = f"{EXECUTABLE} {TEST_DATA_FILE} {PARAM_LINE_NUMBER} "+\
    f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {TEST_HOSPITAL_FILE}"

# Call the model using baseline parameters, pipe output to file, read output file
file_output = open(TEST_OUTPUT_FILE, "w")
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

# # Write df_output to file
df_output.to_csv(TEST_OUTPUT_FILE_HOSPITAL, index = False)


class TestClass(object):
    """
    Test class for checking
    """

    def test_hcw_in_population_list(self):
        """
        Test that healthcare worker IDs correspond to population IDs
        """
        
        df_hcw = pd.read_csv(TEST_HCW_FILE)
        df_population = pd.read_csv(TEST_INDIVIDUAL_FILE)
        hcw_idx_list = df_hcw.pdx.values
        population_idx_list = df_population.ID.values

        assert all(idx in population_idx_list for idx in hcw_idx_list)


    def test_hcw_not_in_work_network(self):
        """
        If worker type not -1, then work network must be -1
        """
        df_interactions = pd.read_csv(TEST_INTERACTIONS_FILE)
        w1_hcw_condition = df_interactions['worker_type_1'] != -1
        w1_worknetwork_condition = df_interactions['work_network'] != -1
        df_test_worker1 = df_interactions[w1_hcw_condition & w1_worknetwork_condition]

        assert len(df_test_worker1.index) == 0
        
        w2_hcw_condition = df_interactions['worker_type_2'] != -1
        w2_worknetwork_condition = df_interactions['work_network_2'] != -1
        df_test_worker2 = df_interactions[w2_hcw_condition & w2_worknetwork_condition]

        assert len(df_test_worker2.index) == 0

    
    def test_hcw_listed_once(self):
        """
        Test that healthcare workers IDs appear only once in the hcw file and therefore only belong to one ward/ hospital
        """
        df_hcw = pd.read_csv(TEST_HCW_FILE)
        hcw_idx_list = df_hcw.pdx.values

        assert len(hcw_idx_list) == len(set(hcw_idx_list))


    def test_ward_capacity(self):
        """
        Test that patients in ward do not exceed ward beds
        """

        df_hcw_time_step = pd.read_csv(TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)
        df_number_beds_exceeded = df_hcw_time_step.query('n_patients > n_beds')

        assert len(df_number_beds_exceeded.index) == 0
























