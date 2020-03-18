#!/usr/bin/env python3
"""
Very basic example to show how testing may work using pytest

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


TEST_DATA_TEMPLATE = "./tests/data/baseline_parameters.csv"
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
    
    
    def test_execution(self):
        """
        Test the code returns an exit code of 0 upon running
        """
        
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("n_total", 50000)
        params.write_params(TEST_DATA_FILE)
        
        # Call the model
        completed_run = subprocess.run([command, TEST_DATA_FILE], capture_output = True)
        
        # Call the model
        np.testing.assert_equal(completed_run.returncode, 0)

