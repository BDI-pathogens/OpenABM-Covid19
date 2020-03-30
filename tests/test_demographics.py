#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Tests of the individual-based model, COVID19-IBM, using the individual file

Created: March 2020
Author: p-robot
"""

"""
Created on Fri Mar 27 13:48:27 2020
Usage:
    pytest -rf tests/test_demographics.py::TestClass::test_demographic_proportions
    from the main folder
@author: anelnurtay
"""


import subprocess, shutil, os
from os.path import join
import numpy as np, pandas as pd
import pytest

from parameters import ParameterSet
import utilities as utils
from math import sqrt
#from test.test_bufio import lengths
#from CoreGraphics._CoreGraphics import CGRect_getMidX

# Directories
IBM_DIR = "src"
IBM_DIR_TEST = "src_test"
DATA_DIR_TEST = "data_test"

TEST_DATA_TEMPLATE = "./tests/data/baseline_parameters.csv"
TEST_DATA_FILE = join(DATA_DIR_TEST, "test_parameters.csv")

TEST_OUTPUT_FILE = join(DATA_DIR_TEST, "test_output.csv")
TEST_INDIVIDUAL_FILE = join(DATA_DIR_TEST, "individual_file_Run1.csv")

TEST_HOUSEHOLD_TEMPLATE = "./tests/data/baseline_household_demographics.csv"
TEST_HOUSEHOLD_FILE = join(DATA_DIR_TEST, "test_household_demographics.csv")

# Age groups
AGE_0     = 0
AGE_10_19 = 1
AGE_20_29 = 2
AGE_30_39 = 3
AGE_40_49 = 4
AGE_50_59 = 5
AGE_60_69 = 6
AGE_70_79 = 7
AGE_80    = 8
AGES = [ AGE_0, AGE_10_19, AGE_20_29, AGE_30_39, AGE_40_49, AGE_50_59, AGE_60_69, AGE_70_79, AGE_80 ]


PARAM_LINE_NUMBER = 1

# Construct the executable command
EXE = "covid19ibm.exe {} {} {} {}".format(TEST_DATA_FILE,
                                       PARAM_LINE_NUMBER,
                                       DATA_DIR_TEST, 
                                       TEST_HOUSEHOLD_FILE)

command = join(IBM_DIR_TEST, EXE)

def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )



class TestClass(object):
    params = {
        "test_file_exists": [dict()],
        "test_demographic_proportions" : [
            dict(
                n_total     = 10000,
                population_0_9  = 10000 * 0.12,
                population_10_19= 10000 * 0.11,
                population_20_29= 10000 * 0.13,
                population_30_39= 10000 * 0.13,
                population_40_49= 10000 * 0.13,
                population_50_59= 10000 * 0.13,
                population_60_69= 10000 * 0.11,
                population_70_79= 10000 * 0.08,
                population_80= 10000 * 0.05
            ),
            dict(
                n_total     = 20000,
                population_0_9  = 20000 * 0.12,
                population_10_19= 20000 * 0.11,
                population_20_29= 20000 * 0.13,
                population_30_39= 20000 * 0.13,
                population_40_49= 20000 * 0.13,
                population_50_59= 20000 * 0.13,
                population_60_69= 20000 * 0.11,
                population_70_79= 20000 * 0.08,
                population_80= 20000 * 0.05
            ),
            dict(
                n_total     = 50000,
                population_0_9  = 50000 * 0.12,
                population_10_19= 50000 * 0.11,
                population_20_29= 50000 * 0.13,
                population_30_39= 50000 * 0.13,
                population_40_49= 50000 * 0.13,
                population_50_59= 50000 * 0.13,
                population_60_69= 50000 * 0.11,
                population_70_79= 50000 * 0.08,
                population_80= 50000 * 0.05
            ),
            dict(
                n_total     = 100000,
                population_0_9  = 100000 * 0.12,
                population_10_19= 100000 * 0.11,
                population_20_29= 100000 * 0.13,
                population_30_39= 100000 * 0.13,
                population_40_49= 100000 * 0.13,
                population_50_59= 100000 * 0.13,
                population_60_69= 100000 * 0.11,
                population_70_79= 100000 * 0.08,
                population_80= 100000 * 0.05
            ),
            dict(
                n_total     = 250000,
                population_0_9  = 250000 * 0.12,
                population_10_19= 250000 * 0.11,
                population_20_29= 250000 * 0.13,
                population_30_39= 250000 * 0.13,
                population_40_49= 250000 * 0.13,
                population_50_59= 250000 * 0.13,
                population_60_69= 250000 * 0.11,
                population_70_79= 250000 * 0.08,
                population_80= 250000 * 0.05
            )
        ]
        }
    """
    Test class for checking 
    """
    @classmethod
    def setup_class(self):
        """
        When the class is instantiated: compile the IBM in a temporary directory
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
        Called before each method is run; creates a new data dir, copies test datasets
        """
        os.mkdir(DATA_DIR_TEST)
        shutil.copy(TEST_DATA_TEMPLATE, TEST_DATA_FILE)
        shutil.copy(TEST_HOUSEHOLD_TEMPLATE, TEST_HOUSEHOLD_FILE)
        
        # Adjust any parameters that need adjusting for all tests
        params = ParameterSet(TEST_DATA_FILE, line_number = 1)
        params.set_param("n_total", 10000)
        params.set_param("end_time", 1)
        params.write_params(TEST_DATA_FILE)
        
    def teardown_method(self):
        """
        At the end of each method (test), remove the directory of test input/output data
        """
        shutil.rmtree(DATA_DIR_TEST, ignore_errors = True)
        
    def test_file_exists(self):
        """
        Test that the individual file exists
        """
        
        # Call the model using baseline parameters, pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command], stdout = file_output, shell = True)
        
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        df_individual = pd.read_csv(TEST_INDIVIDUAL_FILE, comment = "#", sep = ",")
        
        np.testing.assert_equal(df_individual.shape[0] > 1, True)
    
    def test_demographic_proportions(
            self,
            n_total,
            population_0_9,
            population_10_19,
            population_20_29,
            population_30_39,
            population_40_49,
            population_50_59,
            population_60_69,
            population_70_79,
            population_80
        ):
        """
        Test that the proportion of people in different age groups agrees with 
        the population
        """
        
        params = ParameterSet(TEST_DATA_FILE, line_number = 1)
        params.set_param("n_total", n_total)
        params.set_param("n_seed_infection",200)
        params.set_param("end_time", 1)
        params.set_param("infectious_rate", 4.0)  
       
        population_fraction = [ population_0_9,   population_10_19, population_20_29,
                                  population_30_39, population_40_49, population_50_59,
                                  population_60_69, population_70_79, population_80 ]
       
        params.write_params(TEST_DATA_FILE)        
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command], stdout = file_output, shell = True)
        df_indiv = pd.read_csv(TEST_INDIVIDUAL_FILE, comment = "#", sep = ",", skipinitialspace = True )

        # population proportion by age
        N_tot = len( df_indiv )
        for idx in range( len( AGES ) ):
            
            N = len( df_indiv[ ( df_indiv['age_group'] == AGES[idx] ) ] )
            np.testing.assert_allclose( N , population_fraction[idx], atol = N_tot * 0.01)            
