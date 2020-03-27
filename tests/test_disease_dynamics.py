#!/usr/bin/env python3
"""
Tests of the individual-based model, COVID19-IBM, using the individual file

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be 
run by calling 'pytest' from project folder.  

Created: March 2020
Author: p-robot
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
        "test_disease_transition_times": [ 
            dict(
                mean_time_to_symptoms = 4.0,
                sd_time_to_symptoms   = 2.0,
                mean_time_to_hospital = 1.0,
                mean_time_to_critical = 1.0,
                mean_time_to_recover  = 20.0,
                sd_time_to_recover    = 8.0,
                mean_time_to_death    = 12.0,
                sd_time_to_death      = 5.0,
                mean_asymptomatic_to_recovery = 15.0,
                sd_asymptomatic_to_recovery   = 5.0
            ), 
            dict( 
                mean_time_to_symptoms = 4.5,
                sd_time_to_symptoms   = 2.0,
                mean_time_to_hospital = 1.2,
                mean_time_to_critical = 1.2,
                mean_time_to_recover  = 18.0,
                sd_time_to_recover    = 7.0,
                mean_time_to_death    = 14.0,
                sd_time_to_death      = 5.5,
                mean_asymptomatic_to_recovery = 17.0,
                sd_asymptomatic_to_recovery   = 5.0
                
            ),
            dict(
                mean_time_to_symptoms = 5.0,
                sd_time_to_symptoms   = 2.5,
                mean_time_to_hospital = 1.4,
                mean_time_to_critical = 1.4,
                mean_time_to_recover  = 16.0,
                sd_time_to_recover    = 7.0,
                mean_time_to_death    = 16.0,
                sd_time_to_death      = 6,
                mean_asymptomatic_to_recovery = 18.0,
                sd_asymptomatic_to_recovery   = 7.0
            ),
            dict(
                mean_time_to_symptoms = 5.5,
                sd_time_to_symptoms   = 2.5,
                mean_time_to_hospital = 1.6,
                mean_time_to_critical = 1.6,
                mean_time_to_recover  = 14.0,
                sd_time_to_recover    = 6.0,
                mean_time_to_death   = 17.0,
                sd_time_to_death      = 5,
                mean_asymptomatic_to_recovery = 12.0,
                sd_asymptomatic_to_recovery   = 4.0
            ),
            dict(
                mean_time_to_symptoms = 6.0,
                sd_time_to_symptoms   = 3.0,
                mean_time_to_hospital = 1.8,
                mean_time_to_critical = 2.0,
                mean_time_to_recover  = 12.0,
                sd_time_to_recover    = 6.0,
                mean_time_to_death    = 18.0,
                sd_time_to_death      = 6,
                mean_asymptomatic_to_recovery = 14.0,
                sd_asymptomatic_to_recovery   = 5.0
            )
        ] ,
        "test_disease_outcome_proportions" : [
            dict(
                fraction_asymptomatic       = 0.05,
                hospitalised_fraction_0_9   = 0.05,
                hospitalised_fraction_10_19 = 0.10,
                hospitalised_fraction_20_29 = 0.10,
                hospitalised_fraction_30_39 = 0.20,
                hospitalised_fraction_40_49 = 0.20,
                hospitalised_fraction_50_59 = 0.20,
                hospitalised_fraction_60_69 = 0.30,
                hospitalised_fraction_70_79 = 0.30,
                hospitalised_fraction_80    = 0.50,
                critical_fraction_0_9   = 0.05,
                critical_fraction_10_19 = 0.10,
                critical_fraction_20_29 = 0.10,
                critical_fraction_30_39 = 0.20,
                critical_fraction_40_49 = 0.20,
                critical_fraction_50_59 = 0.20,
                critical_fraction_60_69 = 0.30,
                critical_fraction_70_79 = 0.30,
                critical_fraction_80    = 0.50,
                fatality_fraction_0_9   = 0.05,
                fatality_fraction_10_19 = 0.10,
                fatality_fraction_20_29 = 0.10,
                fatality_fraction_30_39 = 0.20,
                fatality_fraction_40_49 = 0.20,
                fatality_fraction_50_59 = 0.20,
                fatality_fraction_60_69 = 0.30,
                fatality_fraction_70_79 = 0.30,
                fatality_fraction_80    = 0.50
            ),
            dict(
                fraction_asymptomatic       = 0.15,
                hospitalised_fraction_0_9   = 0.50,
                hospitalised_fraction_10_19 = 0.40,
                hospitalised_fraction_20_29 = 0.30,
                hospitalised_fraction_30_39 = 0.20,
                hospitalised_fraction_40_49 = 0.20,
                hospitalised_fraction_50_59 = 0.20,
                hospitalised_fraction_60_69 = 0.30,
                hospitalised_fraction_70_79 = 0.30,
                hospitalised_fraction_80    = 0.20,
                critical_fraction_0_9   = 0.50,
                critical_fraction_10_19 = 0.40,
                critical_fraction_20_29 = 0.30,
                critical_fraction_30_39 = 0.20,
                critical_fraction_40_49 = 0.20,
                critical_fraction_50_59 = 0.20,
                critical_fraction_60_69 = 0.30,
                critical_fraction_70_79 = 0.30,
                critical_fraction_80    = 0.20,
                fatality_fraction_0_9   = 0.50,
                fatality_fraction_10_19 = 0.40,
                fatality_fraction_20_29 = 0.30,
                fatality_fraction_30_39 = 0.20,
                fatality_fraction_40_49 = 0.20,
                fatality_fraction_50_59 = 0.20,
                fatality_fraction_60_69 = 0.30,
                fatality_fraction_70_79 = 0.30,
                fatality_fraction_80    = 0.20
            ),
            dict(
                fraction_asymptomatic       = 0.35,
                hospitalised_fraction_0_9   = 0.05,
                hospitalised_fraction_10_19 = 0.05,
                hospitalised_fraction_20_29 = 0.05,
                hospitalised_fraction_30_39 = 0.20,
                hospitalised_fraction_40_49 = 0.20,
                hospitalised_fraction_50_59 = 0.20,
                hospitalised_fraction_60_69 = 0.30,
                hospitalised_fraction_70_79 = 0.80,
                hospitalised_fraction_80    = 0.90,
                critical_fraction_0_9   = 0.05,
                critical_fraction_10_19 = 0.05,
                critical_fraction_20_29 = 0.05,
                critical_fraction_30_39 = 0.20,
                critical_fraction_40_49 = 0.20,
                critical_fraction_50_59 = 0.20,
                critical_fraction_60_69 = 0.30,
                critical_fraction_70_79 = 0.80,
                critical_fraction_80    = 0.90,
                fatality_fraction_0_9   = 0.05,
                fatality_fraction_10_19 = 0.05,
                fatality_fraction_20_29 = 0.05,
                fatality_fraction_30_39 = 0.20,
                fatality_fraction_40_49 = 0.20,
                fatality_fraction_50_59 = 0.20,
                fatality_fraction_60_69 = 0.30,
                fatality_fraction_70_79 = 0.80,
                fatality_fraction_80    = 0.90
            ),
            dict(
                fraction_asymptomatic       = 0.45,
                hospitalised_fraction_0_9   = 0.02,
                hospitalised_fraction_10_19 = 0.02,
                hospitalised_fraction_20_29 = 0.02,
                hospitalised_fraction_30_39 = 0.10,
                hospitalised_fraction_40_49 = 0.15,
                hospitalised_fraction_50_59 = 0.20,
                hospitalised_fraction_60_69 = 0.25,
                hospitalised_fraction_70_79 = 0.30,
                hospitalised_fraction_80    = 0.50,
                critical_fraction_0_9   = 0.02,
                critical_fraction_10_19 = 0.02,
                critical_fraction_20_29 = 0.02,
                critical_fraction_30_39 = 0.10,
                critical_fraction_40_49 = 0.15,
                critical_fraction_50_59 = 0.20,
                critical_fraction_60_69 = 0.25,
                critical_fraction_70_79 = 0.30,
                critical_fraction_80    = 0.50,
                fatality_fraction_0_9   = 0.02,
                fatality_fraction_10_19 = 0.02,
                fatality_fraction_20_29 = 0.02,
                fatality_fraction_30_39 = 0.10,
                fatality_fraction_40_49 = 0.15,
                fatality_fraction_50_59 = 0.20,
                fatality_fraction_60_69 = 0.25,
                fatality_fraction_70_79 = 0.30,
                fatality_fraction_80    = 0.50
            ),
            dict(
                fraction_asymptomatic       = 0.55,
                hospitalised_fraction_0_9   = 0.20,
                hospitalised_fraction_10_19 = 0.20,
                hospitalised_fraction_20_29 = 0.20,
                hospitalised_fraction_30_39 = 0.20,
                hospitalised_fraction_40_49 = 0.20,
                hospitalised_fraction_50_59 = 0.20,
                hospitalised_fraction_60_69 = 0.20,
                hospitalised_fraction_70_79 = 0.20,
                hospitalised_fraction_80    = 0.20,
                critical_fraction_0_9   = 0.20,
                critical_fraction_10_19 = 0.20,
                critical_fraction_20_29 = 0.20,
                critical_fraction_30_39 = 0.20,
                critical_fraction_40_49 = 0.20,
                critical_fraction_50_59 = 0.20,
                critical_fraction_60_69 = 0.20,
                critical_fraction_70_79 = 0.20,
                critical_fraction_80    = 0.20,
                fatality_fraction_0_9   = 0.20,
                fatality_fraction_10_19 = 0.20,
                fatality_fraction_20_29 = 0.20,
                fatality_fraction_30_39 = 0.20,
                fatality_fraction_40_49 = 0.20,
                fatality_fraction_50_59 = 0.20,
                fatality_fraction_60_69 = 0.20,
                fatality_fraction_70_79 = 0.20,
                fatality_fraction_80    = 0.20
            ),
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
        params.set_param("end_time", 100)
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


    def test_disease_transition_times(
            self,
            mean_time_to_symptoms,
            sd_time_to_symptoms,
            mean_time_to_hospital,
            mean_time_to_critical,
            mean_time_to_recover,
            sd_time_to_recover,
            mean_time_to_death,
            sd_time_to_death,
            mean_asymptomatic_to_recovery,
            sd_asymptomatic_to_recovery 
        ):
        """
        Test that the mean and standard deviation of the transition times between 
        states agrees with the parameters
        """
        std_error_limit = 3
        
        params = ParameterSet(TEST_DATA_FILE, line_number = 1)
        params.set_param("n_total", 20000)
        params.set_param("n_seed_infection",200)
        params.set_param("end_time", 50)
        params.set_param("infectious_rate", 4.0)  
        params.set_param("mean_time_to_symptoms",mean_time_to_symptoms)
        params.set_param("sd_time_to_symptoms",sd_time_to_symptoms)
        params.set_param("mean_time_to_hospital",mean_time_to_hospital)
        params.set_param("mean_time_to_critical",mean_time_to_critical)
        params.set_param("mean_time_to_recover",mean_time_to_recover)   
        params.set_param("sd_time_to_recover",sd_time_to_recover)   
        params.set_param("mean_time_to_death",mean_time_to_death)   
        params.set_param("sd_time_to_death",sd_time_to_death)   
        params.set_param("mean_asymptomatic_to_recovery",mean_asymptomatic_to_recovery)   
        params.set_param("sd_asymptomatic_to_recovery",sd_asymptomatic_to_recovery)   
        utils.turn_off_interventions(params,50)
        
        params.write_params(TEST_DATA_FILE)        
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command], stdout = file_output, shell = True)
        df_indiv = pd.read_csv(TEST_INDIVIDUAL_FILE, comment = "#", sep = ",", skipinitialspace = True )
        
        # time infected until showing symptoms
        df_indiv["t_p_s"] =  df_indiv["time_symptomatic"] - df_indiv["time_infected"]  
        mean = df_indiv[ ( df_indiv['time_infected'] > 0 ) & ( df_indiv['time_asymptomatic'] < 0 ) ][ "t_p_s" ].mean()
        sd   = df_indiv[ ( df_indiv['time_infected'] > 0 ) & ( df_indiv['time_asymptomatic'] < 0 ) ][ "t_p_s" ].std()
        N    = len( df_indiv[ ( df_indiv['time_infected'] > 0 ) & ( df_indiv['time_asymptomatic'] < 0 ) ] )
        np.testing.assert_allclose( mean, mean_time_to_symptoms, atol = std_error_limit * sd / sqrt( N ) )
        np.testing.assert_allclose( sd, sd_time_to_symptoms, atol = std_error_limit * sd / sqrt( N ) )
                               
        # time showing symptoms until going to hospital
        df_indiv["t_s_h"] =  df_indiv["time_hospitalised"] - df_indiv["time_symptomatic"]  
        mean = df_indiv[ ( df_indiv['time_hospitalised'] > 0 ) ][ "t_s_h" ].mean()
        sd   = df_indiv[ ( df_indiv['time_hospitalised'] > 0 ) ][ "t_s_h" ].std()
        N    = len( df_indiv[ ( df_indiv['time_hospitalised'] > 0 ) ] )
        np.testing.assert_allclose( mean, mean_time_to_hospital, atol = std_error_limit * sd / sqrt( N ) )
        
        # time hospitalised until moving to the ICU
        df_indiv["t_h_c"] =  df_indiv["time_critical"] - df_indiv["time_hospitalised"]  
        mean = df_indiv[ ( df_indiv['time_critical'] > 0 ) ][ "t_h_c" ].mean()
        sd   = df_indiv[ ( df_indiv['time_critical'] > 0 ) ][ "t_h_c" ].std()
        N    = len( df_indiv[ ( df_indiv['time_critical'] > 0 ) ] )
        np.testing.assert_allclose( mean, mean_time_to_critical, atol = std_error_limit * sd / sqrt( N ) )
        
        # time from symptoms to recover if not hospitalised
        df_indiv["t_s_r"] =  df_indiv["time_recovered"] - df_indiv["time_symptomatic"]  
        mean = df_indiv[ ( df_indiv['time_recovered'] > 0 ) & ( df_indiv['time_asymptomatic'] < 0 ) & ( df_indiv['time_hospitalised'] < 0 ) ][ "t_s_r" ].mean()
        sd   = df_indiv[ ( df_indiv['time_recovered'] > 0 ) & ( df_indiv['time_asymptomatic'] < 0 ) & ( df_indiv['time_hospitalised'] < 0 ) ][ "t_s_r" ].std()
        N    = len( df_indiv[ ( df_indiv['time_recovered'] > 0 ) & ( df_indiv['time_asymptomatic'] < 0 ) & ( df_indiv['time_hospitalised'] < 0 ) ] )
        np.testing.assert_allclose( mean, mean_time_to_recover, atol = std_error_limit * sd / sqrt( N ) )
        np.testing.assert_allclose( sd, sd_time_to_recover, atol = std_error_limit * sd / sqrt( N ) )
       
        # time from hospitalised to recover if don't got to ICU
        df_indiv["t_h_r"] =  df_indiv["time_recovered"] - df_indiv["time_hospitalised"]  
        mean = df_indiv[ ( df_indiv['time_recovered'] > 0 ) & ( df_indiv['time_critical'] < 0 ) & ( df_indiv['time_hospitalised'] > 0 ) ][ "t_h_r" ].mean()
        sd   = df_indiv[ ( df_indiv['time_recovered'] > 0 ) & ( df_indiv['time_critical'] < 0 ) & ( df_indiv['time_hospitalised'] > 0 ) ][ "t_h_r" ].std()
        N    = len( df_indiv[ ( df_indiv['time_recovered'] > 0 ) & ( df_indiv['time_critical'] < 0 ) & ( df_indiv['time_hospitalised'] > 0 ) ] )
        np.testing.assert_allclose( mean, mean_time_to_recover, atol = std_error_limit * sd / sqrt( N ) )
        np.testing.assert_allclose( sd, sd_time_to_recover, atol = std_error_limit * sd / sqrt( N ) )
        
        # time from ICU to recover if don't die
        df_indiv["t_c_r"] =  df_indiv["time_recovered"] - df_indiv["time_critical"]  
        mean = df_indiv[ ( df_indiv['time_recovered'] > 0 ) & ( df_indiv['time_critical'] > 0 ) ][ "t_c_r" ].mean()
        sd   = df_indiv[ ( df_indiv['time_recovered'] > 0 ) & ( df_indiv['time_critical'] > 0 ) ][ "t_c_r" ].std()
        N    = len( df_indiv[ ( df_indiv['time_recovered'] > 0 ) & ( df_indiv['time_critical'] > 0 ) ] )
        np.testing.assert_allclose( mean, mean_time_to_recover, atol = std_error_limit * sd / sqrt( N ) )
        np.testing.assert_allclose( sd, sd_time_to_recover, atol = std_error_limit * sd / sqrt( N ) )
       
        # time from ICU to death
        df_indiv["t_c_d"] =  df_indiv["time_death"] - df_indiv["time_critical"]  
        mean = df_indiv[ ( df_indiv['time_death'] > 0 ) ][ "t_c_d" ].mean()
        sd   = df_indiv[ ( df_indiv['time_death'] > 0 ) ][ "t_c_d" ].std()
        N    = len( df_indiv[ ( df_indiv['time_death'] > 0 ) ] )
        np.testing.assert_allclose( mean, mean_time_to_death, atol = std_error_limit * sd / sqrt( N ) )
        np.testing.assert_allclose( sd, sd_time_to_death, atol = std_error_limit * sd / sqrt( N ) )
       
        # time from asymptomatic to recover
        df_indiv["t_a_r"] =  df_indiv["time_recovered"] - df_indiv["time_asymptomatic"]  
        mean = df_indiv[ ( df_indiv['time_recovered'] > 0 ) & ( df_indiv["time_asymptomatic"] > 0 ) ][ "t_a_r" ].mean()
        sd   = df_indiv[ ( df_indiv['time_recovered'] > 0 ) & ( df_indiv["time_asymptomatic"] > 0 ) ][ "t_a_r" ].std()
        N    = len( df_indiv[ ( df_indiv['time_recovered'] > 0 ) & ( df_indiv["time_asymptomatic"] > 0 ) ] )
        np.testing.assert_allclose( mean, mean_asymptomatic_to_recovery, atol = std_error_limit * sd / sqrt( N ) )
        np.testing.assert_allclose( sd, sd_asymptomatic_to_recovery, atol = std_error_limit * sd / sqrt( N ) )
     
    def test_disease_outcome_proportions(
            self,
            fraction_asymptomatic,
            hospitalised_fraction_0_9,
            hospitalised_fraction_10_19,
            hospitalised_fraction_20_29 ,
            hospitalised_fraction_30_39,
            hospitalised_fraction_40_49,
            hospitalised_fraction_50_59,
            hospitalised_fraction_60_69,
            hospitalised_fraction_70_79,
            hospitalised_fraction_80,
            critical_fraction_0_9,
            critical_fraction_10_19,
            critical_fraction_20_29,
            critical_fraction_30_39,
            critical_fraction_40_49,
            critical_fraction_50_59,
            critical_fraction_60_69,
            critical_fraction_70_79,
            critical_fraction_80,
            fatality_fraction_0_9,
            fatality_fraction_10_19,
            fatality_fraction_20_29,
            fatality_fraction_30_39,
            fatality_fraction_40_49,
            fatality_fraction_50_59,
            fatality_fraction_60_69,
            fatality_fraction_70_79,
            fatality_fraction_80   
        ):
        """
        Test that the fraction of infected people following each path for 
        the progression of the disease agrees with the parameters
        """
        std_error_limit = 3
        
        params = ParameterSet(TEST_DATA_FILE, line_number = 1)
        params.set_param("n_total", 20000)
        params.set_param("n_seed_infection",200)
        params.set_param("end_time", 50)
        params.set_param("infectious_rate", 4.0)  
        params.set_param("fraction_asymptomatic",fraction_asymptomatic)
        params.set_param("hospitalised_fraction_0_9",   hospitalised_fraction_0_9)  
        params.set_param("hospitalised_fraction_10_19", hospitalised_fraction_10_19)  
        params.set_param("hospitalised_fraction_20_29", hospitalised_fraction_20_29)  
        params.set_param("hospitalised_fraction_30_39", hospitalised_fraction_30_39)  
        params.set_param("hospitalised_fraction_40_49", hospitalised_fraction_40_49)  
        params.set_param("hospitalised_fraction_50_59", hospitalised_fraction_50_59)  
        params.set_param("hospitalised_fraction_60_69", hospitalised_fraction_60_69)  
        params.set_param("hospitalised_fraction_70_79", hospitalised_fraction_70_79)  
        params.set_param("hospitalised_fraction_80",    hospitalised_fraction_80)  
        params.set_param("critical_fraction_0_9",   critical_fraction_0_9)  
        params.set_param("critical_fraction_10_19", critical_fraction_10_19)  
        params.set_param("critical_fraction_20_29", critical_fraction_20_29)  
        params.set_param("critical_fraction_30_39", critical_fraction_30_39)  
        params.set_param("critical_fraction_40_49", critical_fraction_40_49)  
        params.set_param("critical_fraction_50_59", critical_fraction_50_59)  
        params.set_param("critical_fraction_60_69", critical_fraction_60_69)  
        params.set_param("critical_fraction_70_79", critical_fraction_70_79)  
        params.set_param("critical_fraction_80",    critical_fraction_80)  
        params.set_param("fatality_fraction_0_9",   fatality_fraction_0_9)  
        params.set_param("fatality_fraction_10_19", fatality_fraction_10_19)  
        params.set_param("fatality_fraction_20_29", fatality_fraction_20_29)  
        params.set_param("fatality_fraction_30_39", fatality_fraction_30_39)  
        params.set_param("fatality_fraction_40_49", fatality_fraction_40_49)  
        params.set_param("fatality_fraction_50_59", fatality_fraction_50_59)  
        params.set_param("fatality_fraction_60_69", fatality_fraction_60_69)  
        params.set_param("fatality_fraction_70_79", fatality_fraction_70_79)  
        params.set_param("fatality_fraction_80",    fatality_fraction_80) 
        utils.turn_off_interventions(params,50)
 
        
        hospitalised_fraction = [ hospitalised_fraction_0_9,   hospitalised_fraction_10_19, hospitalised_fraction_20_29,
                                  hospitalised_fraction_30_39, hospitalised_fraction_40_49, hospitalised_fraction_50_59,
                                  hospitalised_fraction_60_69, hospitalised_fraction_70_79, hospitalised_fraction_80 ]
      
        critical_fraction = [ critical_fraction_0_9,   critical_fraction_10_19, critical_fraction_20_29,
                                  critical_fraction_30_39, critical_fraction_40_49, critical_fraction_50_59,
                                  critical_fraction_60_69, critical_fraction_70_79, critical_fraction_80 ]
      
        fatality_fraction = [ fatality_fraction_0_9,   fatality_fraction_10_19, fatality_fraction_20_29,
                                  fatality_fraction_30_39, fatality_fraction_40_49, fatality_fraction_50_59,
                                  fatality_fraction_60_69, fatality_fraction_70_79, fatality_fraction_80 ]
        
        params.write_params(TEST_DATA_FILE)        
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command], stdout = file_output, shell = True)
        df_indiv = pd.read_csv(TEST_INDIVIDUAL_FILE, comment = "#", sep = ",", skipinitialspace = True )
        
         # fraction asymptomatic vs symptomatc
        N_asym = len( df_indiv[ ( df_indiv['time_infected'] > 0 ) & ( df_indiv['time_asymptomatic'] > 0 ) ] )
        N_sym  = len( df_indiv[ ( df_indiv['time_infected'] > 0 ) & ( df_indiv['time_asymptomatic'] < 0 ) ] )
        N      = N_sym + N_asym
        mean   = N_asym / N
        sd     = sqrt( mean * ( 1 - mean ) )
        np.testing.assert_allclose( mean, fraction_asymptomatic, atol = std_error_limit * sd / sqrt( N ) )
      
        # hospitalised fraction by age
        N_hosp_tot = 0
        N_symp_tot = 0
        hospitalised_fraction_weighted = 0
        for idx in range( len( AGES ) ):
            
            N_hosp = len( df_indiv[ ( df_indiv['time_hospitalised'] > 0 ) & ( df_indiv['age_group'] == AGES[idx] ) ] )
            N_symp = len( df_indiv[ ( df_indiv['time_symptomatic'] >  0 ) & ( df_indiv['age_group'] == AGES[idx] ) ] )
            mean   = N_hosp / N_symp
            sd     = sqrt( mean * ( 1 - mean ) )
            np.testing.assert_allclose( mean, hospitalised_fraction[idx], atol = std_error_limit * sd / sqrt( N_symp ) )
            
            N_hosp_tot += N_hosp
            N_symp_tot += N_symp
            hospitalised_fraction_weighted += hospitalised_fraction[idx] * N_symp

        # overall hospitalised fraction 
        mean   = N_hosp_tot / N_symp_tot
        sd     = sqrt( mean * ( 1 - mean ) )
        hospitalised_fraction_weighted  = hospitalised_fraction_weighted / N_symp_tot
        np.testing.assert_allclose( mean, hospitalised_fraction_weighted, atol = std_error_limit * sd / sqrt( N_symp_tot ) )
          
        # critical fraction by age
        N_crit_tot = 0
        N_hosp_tot = 0
        critical_fraction_weighted = 0
        for idx in range( len( AGES ) ):
            
            N_crit = len( df_indiv[ ( df_indiv['time_critical'] > 0 ) & ( df_indiv['age_group'] == AGES[idx] ) ] )
            N_hosp = len( df_indiv[ ( df_indiv['time_hospitalised'] >  0 ) & ( df_indiv['age_group'] == AGES[idx] ) ] )
            
            if N_hosp > 0:
                mean   = N_crit / N_hosp
                sd     = sqrt( mean * ( 1 - mean ) )
                if N_crit > 0:
                    np.testing.assert_allclose( mean, critical_fraction[idx], atol = std_error_limit * sd / sqrt( N_hosp ) )
            
            N_crit_tot += N_crit
            N_hosp_tot += N_hosp
            critical_fraction_weighted += critical_fraction[idx] * N_hosp

        # overall critical fraction 
        mean   = N_crit_tot / N_hosp_tot
        sd     = sqrt( mean * ( 1 - mean ) )
        critical_fraction_weighted  = critical_fraction_weighted / N_hosp_tot
        np.testing.assert_allclose( mean, critical_fraction_weighted, atol = std_error_limit * sd / sqrt( N_hosp_tot ) )
   
         # critical fraction by age
        N_dead_tot = 0
        N_crit_tot = 0
        fatality_fraction_weighted = 0
        for idx in range( len( AGES ) ):
            
            N_dead = len( df_indiv[ ( df_indiv['time_death'] > 0 ) & ( df_indiv['age_group'] == AGES[idx] ) ] )
            N_crit = len( df_indiv[ ( df_indiv['time_critical'] >  0 ) & ( df_indiv['age_group'] == AGES[idx] ) ] )
           
            if N_crit > 0:
               mean   = N_dead / N_crit
               sd     = sqrt( mean * ( 1 - mean ) )
               if N_dead > 0:
                   np.testing.assert_allclose( mean, fatality_fraction[idx], atol = std_error_limit * sd / sqrt( N_crit ) )
            
            N_dead_tot += N_dead
            N_crit_tot += N_crit
            fatality_fraction_weighted += fatality_fraction[idx] * N_crit

        mean   = N_dead_tot / N_crit_tot
        sd     = sqrt( mean * ( 1 - mean ) )
        fatality_fraction_weighted  = fatality_fraction_weighted / N_crit_tot
        np.testing.assert_allclose( mean, fatality_fraction_weighted, atol = std_error_limit * sd / sqrt( N_crit_tot ) )
   
        
       