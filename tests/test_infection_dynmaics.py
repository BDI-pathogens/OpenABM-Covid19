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
from scipy import optimize

from parameters import ParameterSet
import utilities as utils
from math import sqrt, log, exp
#from test.test_bufio import lengths
#from CoreGraphics._CoreGraphics import CGRect_getMidX

# Directories
IBM_DIR       = "src"
IBM_DIR_TEST  = "src_test"
DATA_DIR_TEST = "data_test"

TEST_DATA_TEMPLATE = "./tests/data/baseline_parameters.csv"
TEST_DATA_FILE     = join(DATA_DIR_TEST, "test_parameters.csv")

TEST_OUTPUT_FILE      = join(DATA_DIR_TEST, "test_output.csv")
TEST_INDIVIDUAL_FILE  = join(DATA_DIR_TEST, "individual_file_Run1.csv")
TEST_INTERACTION_FILE = join(DATA_DIR_TEST, "interactions_Run1.csv")
TEST_TRANSMISSION_FILE = join(DATA_DIR_TEST, "transmission_Run1.csv")

TEST_HOUSEHOLD_TEMPLATE = "./tests/data/baseline_household_demographics.csv"
TEST_HOUSEHOLD_FILE     = join(DATA_DIR_TEST, "test_household_demographics.csv")

# Age groups
AGE_0_9   = 0
AGE_10_19 = 1
AGE_20_29 = 2
AGE_30_39 = 3
AGE_40_49 = 4
AGE_50_59 = 5
AGE_60_69 = 6
AGE_70_79 = 7
AGE_80    = 8
AGES = [ AGE_0_9, AGE_10_19, AGE_20_29, AGE_30_39, AGE_40_49, AGE_50_59, AGE_60_69, AGE_70_79, AGE_80 ]

CHILD   = 0
ADULT   = 1
ELDERLY = 2
AGE_TYPES = [ CHILD, CHILD, ADULT, ADULT, ADULT, ADULT, ADULT, ELDERLY, ELDERLY ]

# network type
HOUSEHOLD = 0
WORK      = 1
RANDOM    = 2

# work networks
NETWORK_0_9   = 0
NETWORK_10_19 = 1
NETWORK_20_69 = 2
NETWORK_70_79 = 3
NETWORK_80    = 4
NETWORKS      = [ NETWORK_0_9, NETWORK_10_19, NETWORK_20_69, NETWORK_70_79, NETWORK_80 ]

# work type networks
NETWORK_CHILD   = 0
NETWORK_ADULT   = 1
NETWORK_ELDERLY = 2
NETWORK_TYPES    = [ NETWORK_CHILD,  NETWORK_ADULT,  NETWORK_ELDERLY]

# infection status
UNINFECTED      = 0
PRESYMPTOMATIC  = 1
ASYMPTOMATIC    = 2
SYMPTOMATIC     = 3
HOSPITALISED    = 4
CRITICAL        = 5

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
        "test_exponential_growth_homogeneous_random": [
            dict(
                n_connections          = 5,
                end_time               = 50,
                infectious_rate        = 3.0,
                mean_infectious_period = 6.0,
                sd_infectious_period   = 2.5,
                n_seed_infection       = 10,
                n_total                = 100000
            ),
            dict(
                n_connections          = 5,
                end_time               = 75,
                infectious_rate        = 2.5,
                mean_infectious_period = 6.0,
                sd_infectious_period   = 2.5,
                n_seed_infection       = 10,
                n_total                = 100000
            ),
            dict(
                n_connections          = 5,
                end_time               = 75,
                infectious_rate        = 2.0,
                mean_infectious_period = 6.0,
                sd_infectious_period   = 2.0,
                n_seed_infection       = 10,
                n_total                = 100000
            ),
            dict(
                n_connections          = 5,
                end_time               = 75,
                infectious_rate        = 3.0,
                mean_infectious_period = 9.0,
                sd_infectious_period   = 3.0,
                n_seed_infection       = 10,
                n_total                = 100000
            ),
            dict(
                n_connections          = 5,
                end_time               = 75,
                infectious_rate        = 3.0,
                mean_infectious_period = 8.0,
                sd_infectious_period   = 8.0,
                n_seed_infection       = 10,
                n_total                = 100000
            ),
        ],
        "test_transmission_pairs": [ 
            dict( 
                n_total         = 50000,
                infectious_rate = 4.0,
                end_time        = 50,
                hospitalised_daily_interactions = 5
            ) 
        ],
        "test_relative_transmission": [
            dict(
                end_time = 250,
                transmission_within = HOUSEHOLD,
                relative_transmission = "relative_transmission_household",
                relative_transmission_value = 0
            ),
            dict(
                end_time = 250,
                transmission_within = HOUSEHOLD,
                relative_transmission = "relative_transmission_household",
                relative_transmission_value = 0.5
            ),
            dict(
                end_time = 250,
                transmission_within = HOUSEHOLD,
                relative_transmission = "relative_transmission_household",
                relative_transmission_value = 1.5
            ),
            dict(
                end_time = 250,
                transmission_within = HOUSEHOLD,
                relative_transmission = "relative_transmission_household",
                relative_transmission_value = 2
            ),
            dict(
                end_time = 250,
                transmission_within = HOUSEHOLD,
                relative_transmission = "relative_transmission_household",
                relative_transmission_value = 5
            ),
            dict(
                end_time = 250,
                transmission_within = WORK,
                relative_transmission = "relative_transmission_workplace",
                relative_transmission_value = 0
            ),
            dict(
                end_time = 250,
                transmission_within = WORK,
                relative_transmission = "relative_transmission_workplace",
                relative_transmission_value = 0.5
            ),
            dict(
                end_time = 250,
                transmission_within = WORK,
                relative_transmission = "relative_transmission_workplace",
                relative_transmission_value = 1.5
            ),
            dict(
                end_time = 250,
                transmission_within = WORK,
                relative_transmission = "relative_transmission_workplace",
                relative_transmission_value = 25
            ),
            dict(
                end_time = 250,
                transmission_within = WORK,
                relative_transmission = "relative_transmission_workplace",
                relative_transmission_value = 5
            ),
            dict(
                end_time = 250,
                transmission_within = RANDOM,
                relative_transmission = "relative_transmission_random",
                relative_transmission_value = 0
            ),
            dict(
                end_time = 250,
                transmission_within = RANDOM,
                relative_transmission = "relative_transmission_random",
                relative_transmission_value = 0.5
            ),
            dict(
                end_time = 250,
                transmission_within = RANDOM,
                relative_transmission = "relative_transmission_random",
                relative_transmission_value = 1.5
            ),
            dict(
                end_time = 250,
                transmission_within = RANDOM,
                relative_transmission = "relative_transmission_random",
                relative_transmission_value = 2
            ),
            dict(
                end_time = 250,
                transmission_within = RANDOM,
                relative_transmission = "relative_transmission_random",
                relative_transmission_value = 5
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
        
        
    def test_transmission_pairs(
        self, 
        n_total,
        end_time,
        infectious_rate,
        hospitalised_daily_interactions
    ):
        
        params = ParameterSet(TEST_DATA_FILE, line_number = 1)
        params = utils.turn_off_interventions( params, end_time)
        params.set_param( "infectious_rate", infectious_rate )
        params.set_param( "n_total", n_total )
        params.set_param( "end_time", end_time )
        params.set_param( "hospitalised_daily_interactions", hospitalised_daily_interactions )
        params.write_params(TEST_DATA_FILE)     

        file_output   = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command], stdout = file_output, shell = True)     
        df_output     = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        df_trans      = pd.read_csv(TEST_TRANSMISSION_FILE, comment = "#", sep = ",", skipinitialspace = True )
 
        # check to see that the number of entries in the transmission file is that in the time-series
        np.testing.assert_equal( len( df_trans ), df_output.loc[ :, "total_infected" ].max(), "length of transmission file is not the number of infected in the time-series" )

        # check to see whether there are transmission from all infected states
        np.testing.assert_equal( sum( df_trans[ "infector_status" ] == PRESYMPTOMATIC ) > 0, True, "no transmission from presymptomatic people" )
        np.testing.assert_equal( sum( df_trans[ "infector_status" ] == SYMPTOMATIC )    > 0, True, "no transmission from symptomatic people" )
        np.testing.assert_equal( sum( df_trans[ "infector_status" ] == ASYMPTOMATIC )   > 0, True, "no transmission from asymptomatic people" )
        np.testing.assert_equal( sum( df_trans[ "infector_status" ] == HOSPITALISED )   > 0, True, "no transmission from hospitalised people" )
        np.testing.assert_equal( sum( df_trans[ "infector_status" ] == CRITICAL )       > 0, True, "no transmission from critical people" )
 
        # check the only people who were infected by someone after 0 time are the seed infections
        np.testing.assert_equal( min( df_trans[ "infector_infected_time" ] ), 0, "the minimum infected time at transmission must be 0 (the seed infection")
        np.testing.assert_equal( len( df_trans[ df_trans[ "infector_infected_time" ] == 0 ] ), int( params.get_param( "n_seed_infection" ) ), "only the seed infection are infected by someone after 0 days" )
        
        # check that some people can get infected after one time step
        np.testing.assert_equal( len( df_trans[ df_trans[ "infector_infected_time" ] == 1 ] ) > 0, True, "nobody is infected by someone who is infected by for one unit of time" )

        # check the maximum time people are stil infections
        max_sd   = 7;
        max_time = float( params.get_param( "mean_infectious_period" ) ) + max_sd * float(  params.get_param( "sd_infectious_period" ) )
        np.testing.assert_equal( max( df_trans[ "infector_infected_time" ] ) < max_time, True, "someone is infectious at a time greater than mean + 7 * std. dev. of the infectious curve " )

        # check that some people are infected across all networks
        np.testing.assert_equal( sum( df_trans[ "infector_network" ] == HOUSEHOLD ) > 0, True, "no transmission on the household network" )
        np.testing.assert_equal( sum( df_trans[ "infector_network" ] == WORK )      > 0, True, "no transmission on the work network" )
        np.testing.assert_equal( sum( df_trans[ "infector_network" ] == RANDOM )    > 0, True, "no transmission on the random network" )

        # check hospitalised people are not transmitting on the work and household networks
        np.testing.assert_equal( sum( ( df_trans[ "infector_network" ] == HOUSEHOLD ) & ( df_trans[ "infector_status" ] == HOSPITALISED ) ), 0, "hospitalised people transmitting on the household network" )
        np.testing.assert_equal( sum( ( df_trans[ "infector_network" ] == WORK ) &      ( df_trans[ "infector_status" ] == HOSPITALISED ) ), 0, "hospitalised people transmitting on the work network" )    

        
    def test_exponential_growth_homogeneous_random(
            self,
            n_connections,
            end_time,
            infectious_rate,
            mean_infectious_period,
            sd_infectious_period,
            n_seed_infection,
            n_total       
        ):
        """
        Test that the exponential growth phase on a homogeneous random network
        ties out with the analyitcal approximation
        """
        
        # calculate the exponential growth by finding rate of growth between
        # fraction_1 and fraction_2 proportion of the population being infected
        fraction_1 = 0.02
        fraction_2 = 0.05
        tolerance  = 0.05
        
        params = ParameterSet(TEST_DATA_FILE, line_number = 1)
        params = utils.set_homogeneous_random_network_only(params,n_connections,end_time)
        params.set_param( "infectious_rate", infectious_rate )
        params.set_param( "mean_infectious_period", mean_infectious_period )
        params.set_param( "sd_infectious_period", sd_infectious_period )
        params.set_param( "n_seed_infection", n_seed_infection ) 
        params.set_param( "n_total", n_total ) 
        params.set_param( "rng_seed", 2 ) 
        params.set_param( "random_interaction_distribution", 0 );
        params.write_params(TEST_DATA_FILE)     
                
        # Call the model using baseline parameters, pipe output to file, read output file
        file_output   = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command], stdout = file_output, shell = True)      
        df_output     = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        df_ts         = df_output.loc[ :, ["Time", "total_infected"]]

        # calculate the rate exponential rate of grwoth from the model
        ts_1  = df_ts[ df_ts[ "total_infected" ] > ( n_total * fraction_1 ) ].min() 
        ts_2  = df_ts[ df_ts[ "total_infected" ] > ( n_total * fraction_2 ) ].min() 
        slope = ( log( ts_2[ "total_infected"]) - log( ts_1[ "total_infected"]) ) / ( ts_2[ "Time" ] - ts_1[ "Time" ] )
        
        # this is an analytical approximation in the limit of large 
        # mean_infectious_rate, but works very well for most values
        # unless there is large amount infection the following day
        theta     = sd_infectious_period * sd_infectious_period / mean_infectious_period
        k         = mean_infectious_period / theta
        char_func = lambda x: exp(x) - infectious_rate / ( 1 + x * theta )**k
        slope_an  = optimize.brentq( char_func, - 0.99 / theta, 1  )
        
        np.testing.assert_allclose( slope, slope_an, rtol = tolerance, err_msg = "exponential growth deviates too far from analytic approximation")


        
    def test_relative_transmission(
            self,
            end_time,
            transmission_within,
            relative_transmission,
            relative_transmission_value
        ):
        """
        Test that monotonic change in relative_transmission_NETWORK
        leads to corresponding change in counts of the within network transmissions.
        
        The relative_transmission rates r0, r1, r2 produce n0, n1, n2 transmissions, 
        whereas rates k*r0, r1, r2 produce m0, m1, m2 transmissions.
        This test checks that (k * n0 / (k * n0 + n1 + n2) ) is close enough to ( m0 / (m0 + m1 + m2)) 
        for k = 0, 0.5, 1.5, 2, 5 for i = 0, 1, 2 (i = 0 shown in the example).
        """
        tolerance = 0.1

        params = ParameterSet(TEST_DATA_FILE, line_number = 1)
#       Run for even relative transmissions
        params.set_param( "end_time", end_time )
        params.set_param( "relative_transmission_household", 1 )
        params.set_param( "relative_transmission_workplace", 1 )
        params.set_param( "relative_transmission_random", 1 )
        params.write_params(TEST_DATA_FILE)     

        file_output   = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command], stdout = file_output, shell = True)     
        df_trans_even      = pd.read_csv(TEST_TRANSMISSION_FILE, comment = "#", sep = ",", skipinitialspace = True )
        
        # calculating the weighted ratio
        len_household = len( df_trans_even[ df_trans_even[ "infector_network" ] == HOUSEHOLD ] )
        len_work = len( df_trans_even[ df_trans_even[ "infector_network" ] == WORK ] ) 
        len_random = len( df_trans_even[ df_trans_even[ "infector_network" ] == RANDOM ] )
        lengths = [int(len_household), int(len_work), int(len_random)]
        lengths[transmission_within] = lengths[transmission_within] * relative_transmission_value
        all_trans_even = sum(lengths)
        ratio_even = float( df_trans_even[ df_trans_even[ "infector_network" ] == transmission_within ].shape[0] ) * relative_transmission_value / float(all_trans_even) 
        
#       Run for the scaled relative transmission
        params.set_param(relative_transmission , relative_transmission_value )
        params.write_params(TEST_DATA_FILE)     

        file_output   = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command], stdout = file_output, shell = True)     
        df_trans      = pd.read_csv(TEST_TRANSMISSION_FILE, comment = "#", sep = ",", skipinitialspace = True )
        
        all_trans = len( df_trans[ df_trans[ "infector_network" ] == HOUSEHOLD ] ) + \
                    len( df_trans[ df_trans[ "infector_network" ] == WORK ] ) + \
                    len( df_trans[ df_trans[ "infector_network" ] == RANDOM ] )
        ratio_new = float( df_trans[ df_trans[ "infector_network" ] == transmission_within ].shape[0] ) / float(all_trans)

        # check the proportion of the infections
        np.testing.assert_allclose( ratio_new , ratio_even, atol = tolerance)