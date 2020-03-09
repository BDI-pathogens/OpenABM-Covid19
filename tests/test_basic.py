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
import numpy as np

# Directories
IBM_DIR = "src"
IBM_DIR_TEST = IBM_DIR + "_test"
EXE = "covid19ibm.exe"

# Construct the executable command
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
    
    def test_execution(self):
        """
        Test the code returns an exit code of 0 upon running
        """
        
        # Call the model
        completed_run = subprocess.run([command, "./tests/data/test_parameters.csv"], 
            capture_output = True)
        print(completed_run)
        np.testing.assert_equal(completed_run.returncode, 0)

