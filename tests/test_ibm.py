#!/usr/bin/env python3
"""
Tests of the individual-based model, COVID19-IBM

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be 
run by calling 'pytest' from project folder.  

Created: March 2020
Author: p-robot
"""

import subprocess, shutil, os
from os.path import join
import numpy as np, pandas as pd

from parameters import ParameterSet

# Directories
IBM_DIR = "src"
IBM_DIR_TEST = "src_test"
DATA_DIR_TEST = "data_test"


TEST_DATA_TEMPLATE = "./tests/data/test_parameters.csv"
TEST_DATA_FILE = join(DATA_DIR_TEST, "test_parameters.csv")
TEST_OUTPUT_FILE = join(DATA_DIR_TEST, "test_output.csv")

# Construct the executable command
EXE = "covid19ibm.exe"
command = join(IBM_DIR_TEST, EXE)

class TestClass(object):
    """
    Test class for checking 
    """
    @classmethod
    def setup_class(self):
        """
        When the class is instatiated: compile the IBM in a temporary directory
        """
        
        # Make a temporary copy of the code (remove this temporary directory if it already exists)
        shutil.rmtree(IBM_DIR_TEST, ignore_errors = True)
        shutil.copytree(IBM_DIR, IBM_DIR_TEST)
                
        # Construct the compilation command and compile
        compile_command = "make clean; make all"
        completed_compilation = subprocess.run([compile_command], 
            shell = True, cwd = IBM_DIR_TEST, capture_output = True)
    
    @classmethod
    def teardown_class(self):
        """
        Remove the temporary code directory (when this class is removed)
        """
        shutil.rmtree(IBM_DIR_TEST, ignore_errors = True)
    
    def setup_method(self):
        """
        
        """
        os.mkdir(DATA_DIR_TEST)
        shutil.copy(TEST_DATA_TEMPLATE, TEST_DATA_FILE)
        
    def teardown_method(self):
        """
        
        """
        shutil.rmtree(DATA_DIR_TEST, ignore_errors = True)
    
    def test_total_infected_non_negative(self):
        """
        Test tthat he total_infected column should be positive.
        """
        df_params = pd.read_csv(TEST_DATA_FILE)

        # Call the model
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE], stdout = file_output)

        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",", nrows = 50)
        np.testing.assert_equal(np.sum(df_output.total_infected < 0), 0)


    def test_total_infectious_rate_zero(self):
        """
        Test setting infectious rate to zero
        """
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("infectious_rate", 0.0)
        params.write_params(TEST_DATA_FILE)


        # Call the model
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE], stdout = file_output)

        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",", nrows = 50)

        np.testing.assert_equal(np.sum(df_output.total_infected > 0), 0)


    def test_mean_time_to_death_zero(self):
        """
        Test setting mean_time_to_death to zero
        """
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("mean_time_to_death", 0.0)
        params.set_param("sd_time_to_death", 0.0)
        params.write_params(TEST_DATA_FILE)


        # Call the model
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE], stdout = file_output)

        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",", nrows = 50)
        np.testing.assert_equal(np.sum(df_output.total_infected > 0), 0)


    def test_mean_daily_interactions_zero(self):
        """
        Test setting mean_daily_interactions to zero
        """
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("mean_daily_interactions", 0)
        params.write_params(TEST_DATA_FILE)
        print(params.params)
        
        # Call the model
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE], stdout = file_output)
        
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",", nrows = 50)
        print(df_output)
        np.testing.assert_equal(np.sum(df_output.total_infected > 0), 0)

