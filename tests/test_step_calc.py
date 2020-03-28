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

import sys
import subprocess, shutil, os
from os.path import join
import numpy as np, pandas as pd
from random import randrange

sys.path.append("python/")
from model import Model


# STEPS > 0
STEPS = randrange(1, 10)
PARAM_LINE_NUMBER = 1

# Directories
IBM_DIR = "src"
IBM_DIR_TEST = "src_test"
DATA_DIR_TEST = "data_test"

TEST_DATA_TEMPLATE = "./tests/data/baseline_parameters.csv"
TEST_DATA_HOUSEHOLD_DEMOGRAPHICS = "./tests/data/baseline_household_demographics.csv"
TEST_DATA_FILE = join(DATA_DIR_TEST, "test_params.csv")


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
        shutil.rmtree(IBM_DIR_TEST, ignore_errors=True)
        shutil.copytree(IBM_DIR, IBM_DIR_TEST)

        # Construct the compilation command and compile
        compile_command = "make clean; make swig-all"
        completed_compilation = subprocess.run(
            [compile_command], shell=True, cwd=IBM_DIR_TEST, capture_output=True
        )

    @classmethod
    def teardown_class(self):
        """
        Remove the temporary code directory (when this class is removed)
        """
        shutil.rmtree(IBM_DIR_TEST, ignore_errors=True)

    def setup_method(self):
        """

        """
        os.mkdir(DATA_DIR_TEST)
        shutil.copy(TEST_DATA_TEMPLATE, TEST_DATA_FILE)

    def teardown_method(self):
        """

        """
        shutil.rmtree(DATA_DIR_TEST, ignore_errors=True)

    def test_basic_step_calculation(self):
        """
        Test the a parameter can be changed in between step runs
        """
        # Create model object
        model = Model(
            TEST_DATA_TEMPLATE,
            PARAM_LINE_NUMBER,
            DATA_DIR_TEST,
            TEST_DATA_HOUSEHOLD_DEMOGRAPHICS,
        )
        step_model = model.create()

        # Run steps
        for step in range(0, STEPS):
            model.one_time_step(step_model)
            print(model.one_time_step_results(step_model))

            model.set_param("test_on_symptoms", step)
            np.testing.assert_equal(model.get_param("test_on_symptoms"), step)

        model.write_output_files()

        # Destroy the model
        model.destroy(step_model)
