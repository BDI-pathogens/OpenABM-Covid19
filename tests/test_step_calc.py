#!/usr/bin/env python3
"""
Very basic example to show how testing may work using pytest

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be 
run by calling 'pytest' from project folder.

pytest ./tests/test_step_calc.py

Created: March 2020
Author: Daniel Montero
"""

import subprocess, shutil, os
from os.path import join
import numpy as np, pandas as pd
import pexpect

from parameters import ParameterSet

#FIXME these should be random
STEPS = 2
PARAM_LINE_NUMBER = 1

# Directories
IBM_DIR = "src"
IBM_DIR_TEST = "src_test"
DATA_DIR_TEST = "data_test"

TEST_DATA_TEMPLATE = "./tests/data/baseline_parameters.csv"
TEST_DATA_FILE = join(DATA_DIR_TEST, "test_params.csv")
TEST_OUTPUT_FILE = "individual_file_Run{}.csv".format(PARAM_LINE_NUMBER)
TEST_OUTPUT_FILE = join(DATA_DIR_TEST, TEST_OUTPUT_FILE)

# List of temporary parameters files
tmp_param_files = [
                   "{}/tmp_param_file_{}.csv".format(DATA_DIR_TEST, num)
                   for num in range(1, STEPS)
                  ]

# Construct the executable command
EXE = "covid19ibm.exe {} {} {}".format(TEST_DATA_FILE,
                                       PARAM_LINE_NUMBER,
                                       DATA_DIR_TEST)
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
        
        # Make a temporary copy of the code (remove this temporary directory if
        # it already exists)
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
    
    
    def test_basic_step_calculation(self):
        """
        Test the code returns an exit code of 0 upon running
        """
        
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("quarantine_days", 4)
        params.write_params(TEST_DATA_FILE)
        
        # Call the model
        print("command: {}".format(command))
        print(tmp_param_files)
        child = pexpect.spawn(command)

        for step in range(1, STEPS):
            # Wait for results of step x to be ready
            result = child.expect("New params file>")
        
            # Read results
            df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
            # FIXME: check for some value
            #print(df_output)
            #np.testing.assert_equal(df_output["total_infected"].sum(), 0) 
        
            # Create new parameters file
            quarantine_days = int(params.get_param("quarantine_days")) + 1
            params.set_param("quarantine_days", quarantine_days)
            params.write_params(tmp_param_files[step-1])
            
            print(tmp_param_files[step-1])
            child.sendline(tmp_param_files[step-1])
            result = child.expect("OK")
            output = child.readline()
            print("Result: {}".format(result))
            print("Output: {}". format(output))

        # FIXME Test C program's result
        #np.testing.assert_equal(completed_run.returncode, 0)
