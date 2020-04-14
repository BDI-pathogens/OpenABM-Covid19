#!/usr/bin/env python3
"""
Tests of the individual-based model, COVID19-IBM

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be 
run by calling 'pytest' from project folder.  

Created: April 2020
Author: feldnerd
"""

import subprocess
import numpy as np, pandas as pd
import pytest

# File paths/ command line parameters
TEST_DATA_FILE = "/Users/feldnerd/Documents/GitHub/COVID19-IBM/tests/data/baseline_parameters.csv"
PARAM_LINE_NUMBER = 1
DATA_DIR_TEST = "/Users/feldnerd/Documents/GitHub/COVID19-IBM/tests/data"
TEST_HOUSEHOLD_FILE = "/Users/feldnerd/Documents/GitHub/COVID19-IBM/tests/data/baseline_household_demographics.csv"
TEST_HOSPITAL_FILE = "/Users/feldnerd/Documents/GitHub/COVID19-IBM/tests/data/hospital_baseline_parameters.csv"
TEST_OUTPUT_FILE = "/Users/feldnerd/Documents/GitHub/COVID19-IBM/tests/data/test_output.csv"
TEST_OUTPUT_FILE_HOSPITAL = "/Users/feldnerd/Documents/GitHub/COVID19-IBM/tests/data/test_hospital_output.csv"
EXECUTABLE = "/Users/feldnerd/Documents/GitHub/COVID19-IBM/src/covid19ibm.exe"


# Construct the executable command
EXE = f"{EXECUTABLE} {TEST_DATA_FILE} {PARAM_LINE_NUMBER} "+\
    f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {TEST_HOSPITAL_FILE}"

print(EXE)

class TestClass(object):
    """
    Test class for checking
    """

    def test_hospital(self):
        """
        Test that all columns are non-negative
        """
        
        # Call the model using baseline parameters, pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        # Assert output not empty
        assert df_output.size > 1