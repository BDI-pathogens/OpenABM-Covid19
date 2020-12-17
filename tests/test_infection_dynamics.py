#!/usr/bin/env python3
"""
Tests of the individual-based model, COVID19-IBM, using the individual file

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be 
run by calling 'pytest' from project folder.  

Created: March 2020
Author: p-robot
"""

import pytest, sys, subprocess, shutil, os
import numpy as np, pandas as pd
from scipy import optimize
from math import exp, log, fabs

sys.path.append("src/COVID19")
from parameters import ParameterSet

from . import constant
from . import utilities as utils

#from test.test_bufio import lengths
#from CoreGraphics._CoreGraphics import CGRect_getMidX


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )


# Override setup_covid_methods() in conftest.py
@pytest.fixture(scope = "function")
def setup_covid_methods(request):
    """
    Called before each method is run; creates a new data dir, copies test datasets
    """
    os.mkdir(constant.DATA_DIR_TEST)
    shutil.copy(constant.TEST_DATA_TEMPLATE, constant.TEST_DATA_FILE)
    shutil.copy(constant.TEST_HOUSEHOLD_TEMPLATE, constant.TEST_HOUSEHOLD_FILE)
    shutil.copy(constant.TEST_HOSPITAL_TEMPLATE, constant.TEST_HOSPITAL_FILE)

    # Adjust any parameters that need adjusting for all tests
    params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
    params.set_param("n_total", 10000)
    params.set_param("end_time", 1)
    params.write_params(constant.TEST_DATA_FILE)
    def fin():
        """
        At the end of each method (test), remove the directory of test input/output data
        """
        shutil.rmtree(constant.DATA_DIR_TEST, ignore_errors=True)
    request.addfinalizer(fin)

       
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
                n_connections=5,
                end_time=75,
                infectious_rate=2.0,
                mean_infectious_period=6.0,
                sd_infectious_period=2.0,
                n_seed_infection=10,
                n_total=100000
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
                sd_infectious_period   = 7,
                n_seed_infection       = 10,
                n_total                = 100000
            ),
        ],
        "test_transmission_pairs": [
            dict( 
                n_total         = 200000,
                infectious_rate = 8,
                end_time        = 150,
                hospitalised_daily_interactions = 5,
                mean_infectious_period=8.0,
                sd_infectious_period=5,
                mean_time_to_hospital=2,
                mean_time_to_critical=2,
                mean_time_to_symptoms=2
            )
        ],
        "test_monoton_relative_transmission": [
            dict(
                end_time = 250,
                transmission_NETWORK = constant.HOUSEHOLD,
                relative_transmission_values = [0, 0.5, 1, 1.5, 2, 10, 100]
            ),
            dict(
                end_time = 250,
                transmission_NETWORK = constant.OCCUPATION,
                relative_transmission_values = [0, 0.5, 1, 1.5, 2, 10, 100]
            ),
            dict(
                end_time = 250,
                transmission_NETWORK = constant.RANDOM,
                relative_transmission_values = [0, 0.5, 1, 1.5, 2, 10, 100]
            ),
            dict( # fluctuating list
                end_time = 250,
                transmission_NETWORK = constant.OCCUPATION,
                relative_transmission_values = [1.1, 1, 0, 0.1, 0.1, 0.1, 0.3]
            )
        ],
        "test_monoton_fraction_asymptomatic": [
            dict(
                end_time = 250,
                fraction_asymptomatic_0_9 = [0, 0.2, 0.5, 0.5, 1, 0.1],
                fraction_asymptomatic_10_19 = [0, 0.2, 0.5, 0.5, 1, 0.1],
                fraction_asymptomatic_20_29 = [0, 0.2, 0.5, 0.5, 1, 0.1],
                fraction_asymptomatic_30_39 = [0, 0.2, 0.5, 0.5, 1, 0.1],
                fraction_asymptomatic_40_49 = [0, 0.2, 0.5, 0.5, 1, 0.1],
                fraction_asymptomatic_50_59 = [0, 0.2, 0.5, 0.5, 1, 0.1],
                fraction_asymptomatic_60_69 = [0, 0.2, 0.5, 0.5, 1, 0.1],
                fraction_asymptomatic_70_79 = [0, 0.2, 0.5, 0.5, 1, 0.1],
                fraction_asymptomatic_80 = [0, 0.2, 0.5, 0.5, 1, 0.1]
            )        
        ],
        "test_monoton_asymptomatic_infectious_factor": [
            dict(
                end_time = 250,
                asymptomatic_infectious_factor = [0, 0.25, 0.5, 0.5, 1, 0.1]
            )
        ],
        "test_monoton_relative_susceptibility": [
            dict(
                end_time = 250,
                relative_susceptibility_0_9 = [0, 0.5, 1, 1.5, 2],
                relative_susceptibility_10_19 = [1, 1, 1, 1, 1],
                relative_susceptibility_20_29 = [1, 1, 1, 1, 1],
                relative_susceptibility_30_39 = [1, 1, 1, 1, 1],
                relative_susceptibility_40_49 = [1, 1, 1, 1, 1],
                relative_susceptibility_50_59 = [1, 1, 1, 1, 1],
                relative_susceptibility_60_69 = [1, 1, 1, 1, 1],
                relative_susceptibility_70_79 = [1, 1, 1, 1, 1],
                relative_susceptibility_80 = [1, 1, 1, 1, 1]
            ),
            dict(
                end_time = 250,
                relative_susceptibility_0_9 = [1, 1, 1, 1, 1],
                relative_susceptibility_10_19 = [0, 0.5, 1, 1.5, 2],
                relative_susceptibility_20_29 = [1, 1, 1, 1, 1],
                relative_susceptibility_30_39 = [1, 1, 1, 1, 1],
                relative_susceptibility_40_49 = [1, 1, 1, 1, 1],
                relative_susceptibility_50_59 = [1, 1, 1, 1, 1],
                relative_susceptibility_60_69 = [1, 1, 1, 1, 1],
                relative_susceptibility_70_79 = [1, 1, 1, 1, 1],
                relative_susceptibility_80 = [1, 1, 1, 1, 1]
            ),
            dict(
                end_time = 250,
                relative_susceptibility_0_9 = [1, 1, 1, 1, 1],
                relative_susceptibility_10_19 = [1, 1, 1, 1, 1],
                relative_susceptibility_20_29 = [0, 0.5, 1, 1.5, 2],
                relative_susceptibility_30_39 = [1, 1, 1, 1, 1],
                relative_susceptibility_40_49 = [1, 1, 1, 1, 1],
                relative_susceptibility_50_59 = [1, 1, 1, 1, 1],
                relative_susceptibility_60_69 = [1, 1, 1, 1, 1],
                relative_susceptibility_70_79 = [1, 1, 1, 1, 1],
                relative_susceptibility_80 = [1, 1, 1, 1, 1]
            ),
            dict(
                end_time = 250,
                relative_susceptibility_0_9 = [1, 1, 1, 1, 1],
                relative_susceptibility_10_19 = [1, 1, 1, 1, 1],
                relative_susceptibility_20_29 = [1, 1, 1, 1, 1],
                relative_susceptibility_30_39 = [0, 0.5, 1, 1.5, 2],
                relative_susceptibility_40_49 = [1, 1, 1, 1, 1],
                relative_susceptibility_50_59 = [1, 1, 1, 1, 1],
                relative_susceptibility_60_69 = [1, 1, 1, 1, 1],
                relative_susceptibility_70_79 = [1, 1, 1, 1, 1],
                relative_susceptibility_80 = [1, 1, 1, 1, 1]
            ),
            dict(
                end_time = 250,
                relative_susceptibility_0_9 = [1, 1, 1, 1, 1],
                relative_susceptibility_10_19 = [1, 1, 1, 1, 1],
                relative_susceptibility_20_29 = [1, 1, 1, 1, 1],
                relative_susceptibility_30_39 = [1, 1, 1, 1, 1],
                relative_susceptibility_40_49 = [0, 0.5, 1, 1.5, 2],
                relative_susceptibility_50_59 = [1, 1, 1, 1, 1],
                relative_susceptibility_60_69 = [1, 1, 1, 1, 1],
                relative_susceptibility_70_79 = [1, 1, 1, 1, 1],
                relative_susceptibility_80 = [1, 1, 1, 1, 1]
            ),
            dict(
                end_time = 250,
                relative_susceptibility_0_9 = [1, 1, 1, 1, 1],
                relative_susceptibility_10_19 = [1, 1, 1, 1, 1],
                relative_susceptibility_20_29 = [1, 1, 1, 1, 1],
                relative_susceptibility_30_39 = [1, 1, 1, 1, 1],
                relative_susceptibility_40_49 = [1, 1, 1, 1, 1],
                relative_susceptibility_50_59 = [0, 0.5, 1, 1.5, 2],
                relative_susceptibility_60_69 = [1, 1, 1, 1, 1],
                relative_susceptibility_70_79 = [1, 1, 1, 1, 1],
                relative_susceptibility_80 = [1, 1, 1, 1, 1]
            ),
            dict(
                end_time = 250,
                relative_susceptibility_0_9 = [1, 1, 1, 1, 1],
                relative_susceptibility_10_19 = [1, 1, 1, 1, 1],
                relative_susceptibility_20_29 = [1, 1, 1, 1, 1],
                relative_susceptibility_30_39 = [1, 1, 1, 1, 1],
                relative_susceptibility_40_49 = [1, 1, 1, 1, 1],
                relative_susceptibility_50_59 = [1, 1, 1, 1, 1],
                relative_susceptibility_60_69 = [0, 0.5, 1, 1.5, 2],
                relative_susceptibility_70_79 = [1, 1, 1, 1, 1],
                relative_susceptibility_80 = [1, 1, 1, 1, 1]
            ),
            dict(
                end_time = 250,
                relative_susceptibility_0_9 = [1, 1, 1, 1, 1],
                relative_susceptibility_10_19 = [1, 1, 1, 1, 1],
                relative_susceptibility_20_29 = [1, 1, 1, 1, 1],
                relative_susceptibility_30_39 = [1, 1, 1, 1, 1],
                relative_susceptibility_40_49 = [1, 1, 1, 1, 1],
                relative_susceptibility_50_59 = [1, 1, 1, 1, 1],
                relative_susceptibility_60_69 = [1, 1, 1, 1, 1],
                relative_susceptibility_70_79 = [0, 0.5, 1, 1.5, 2],
                relative_susceptibility_80 = [1, 1, 1, 1, 1]
            ),
            dict(
                end_time = 250,
                relative_susceptibility_0_9 = [1, 1, 1, 1, 1],
                relative_susceptibility_10_19 = [1, 1, 1, 1, 1],
                relative_susceptibility_20_29 = [1, 1, 1, 1, 1],
                relative_susceptibility_30_39 = [1, 1, 1, 1, 1],
                relative_susceptibility_40_49 = [1, 1, 1, 1, 1],
                relative_susceptibility_50_59 = [1, 1, 1, 1, 1],
                relative_susceptibility_60_69 = [1, 1, 1, 1, 1],
                relative_susceptibility_70_79 = [1, 1, 1, 1, 1],
                relative_susceptibility_80 = [0, 0.5, 1, 1.5, 2]
            )
        ],
        "test_monoton_mild_infectious_factor": [
            dict(
                end_time = 250,
                mild_fraction_0_9 = 0.8,
                mild_fraction_10_19= 0.8,
                mild_fraction_20_29= 0.8,
                mild_fraction_30_39= 0.8,
                mild_fraction_40_49= 0.8,
                mild_fraction_50_59= 0.8,
                mild_fraction_60_69= 0.8,
                mild_fraction_70_79= 0.8,
                mild_fraction_80= 0.8,
                mild_infectious_factor = [0.1, 0.25, 0.5, 0.5, 1, 0.1]
            )        
        ],
        "test_ratio_presymptomatic_symptomatic": [
            dict(
                n_total = 10000,
                n_seed_infection = 1,
                end_time = 100
            ),
            dict(
                n_total = 100000,
                n_seed_infection = 10,
                end_time = 1
            ),
            dict(
                n_total = 1000000,
                n_seed_infection = 100,
                end_time = 1
            ),
            dict(
                n_total = 100000,
                n_seed_infection = 10000,
                end_time = 1
            ),
        ],
        "test_relative_transmission_update": [
            dict(
                test_params = dict(
                    n_total = 100000,
                    n_seed_infection = 100,
                    infectious_rate = 5.0,
                    relative_transmission_household = 1,
                    relative_transmission_occupation = 1,
                    relative_transmission_random = 1,          
                    end_time = 20,
                    hospital_on = 0
                ),
                update_relative_transmission_household = 0.9,
                update_relative_transmission_occupation = 0.8,
                update_relative_transmission_random = 0.8,          
            ),
        ],
        "test_single_seed_infections": [
            dict(
                test_params = dict(
                    n_total = 1e4,
                    n_seed_infection = 5e3,
                    infectious_rate = 0.0
                ),
            ),
        ],
        "test_presymptomatic_symptomatic_transmissions": [
            dict(
                n_total = 500000,
                n_seed_infection = 1,
                end_time = 100
            ),
            dict(
                n_total = 250000,
                n_seed_infection = 1,
                end_time = 100
            ),
            dict(
                n_total = 1000000,
                n_seed_infection = 1,
                end_time = 100
            )
        ],
        "test_infectiousness_multiplier": [
            dict(
                test_params = dict(
                    n_total = 1e4,
                    n_seed_infection = 50,
                    end_time = 30,
                ),
                sd_multipliers = [0, 0.25, 0.5],
            )
        ],
        "test_infectiousness_multiplier_transmissions_increase_with_multiplier": [
            dict(
                test_params = dict(
                    n_total = 1e4,
                    n_seed_infection = 10,
                    end_time = 50,
                    sd_infectiousness_multiplier = 0.5,
                ),
                n_bins = 5,
            )
        ],
    }
    """
    Test class for checking 
    """
    def test_transmission_pairs(
        self, 
        n_total,
        end_time,
        infectious_rate,
        hospitalised_daily_interactions,
        mean_infectious_period,
        sd_infectious_period,
        mean_time_to_hospital,
        mean_time_to_critical,
        mean_time_to_symptoms
    ):
        
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params = utils.turn_off_interventions( params, end_time)
        params.set_param( "infectious_rate", infectious_rate )
        params.set_param( "n_total", n_total )
        params.set_param( "end_time", end_time )
        params.set_param( "hospitalised_daily_interactions", hospitalised_daily_interactions )
        params.set_param( "mild_infectious_factor", 1)
        params.set_param( "asymptomatic_infectious_factor", 1)
        params.set_param("mean_infectious_period", mean_infectious_period)
        params.set_param("sd_infectious_period", sd_infectious_period)
        params.set_param("mean_time_to_hospital", mean_time_to_hospital)
        params.set_param("mean_time_to_critical", mean_time_to_critical)
        params.set_param("mean_time_to_symptoms", mean_time_to_symptoms)
        params.write_params(constant.TEST_DATA_FILE)     

        file_output   = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)     
        df_output     = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
        df_trans      = pd.read_csv(constant.TEST_TRANSMISSION_FILE, 
            comment = "#", sep = ",", skipinitialspace = True )
        # check to see that the number of entries in the transmission file is that in the time-series
        np.testing.assert_equal( len( df_trans ), df_output.loc[ :, "total_infected" ].max(), "length of transmission file is not the number of infected in the time-series" )

        # check to see whether there are transmission from all infected states
        np.testing.assert_equal( len( df_trans[ df_trans[ "status_source" ] == constant.EVENT_TYPES.PRESYMPTOMATIC.value] ) > 0, True, "no transmission from presymptomatic people" )
        np.testing.assert_equal( len( df_trans[ df_trans[ "status_source" ] == constant.EVENT_TYPES.PRESYMPTOMATIC_MILD.value] ) > 0, True, "no transmission from presymptomatic (mild) people" )
        np.testing.assert_equal( len( df_trans[ df_trans[ "status_source" ] == constant.EVENT_TYPES.SYMPTOMATIC.value] ) > 0 , True, "no transmission from symptomatic people" )
        np.testing.assert_equal( len( df_trans[ df_trans[ "status_source" ] == constant.EVENT_TYPES.SYMPTOMATIC_MILD.value] ) > 0, True, "no transmission from symptomatic (mild) people" )
        np.testing.assert_equal( len( df_trans[ df_trans[ "status_source" ] == constant.EVENT_TYPES.ASYMPTOMATIC.value] )  > 0, True, "no transmission from asymptomatic people" )
        np.testing.assert_equal( len( df_trans[ df_trans[ "status_source" ] == constant.EVENT_TYPES.HOSPITALISED.value] )  > 0, True, "no transmission from hospitalised people" )
        np.testing.assert_equal( len( df_trans[ df_trans[ "status_source" ] == constant.EVENT_TYPES.CRITICAL.value] )      > 0, True, "no transmission from critical people" )
 
        # check the only people who were infected by someone after 0 time are the seed infections
        np.testing.assert_equal( min( df_trans[ "generation_time" ] ), 0, "the minimum infected time at transmission must be 0 (the seed infection")
        np.testing.assert_equal( len( df_trans[ df_trans[ "generation_time" ] == 0 ] ), int( params.get_param( "n_seed_infection" ) ), "only the seed infection are infected by someone after 0 days" )
        
        # check that some people can get infected after one time step
        np.testing.assert_equal( len( df_trans[ df_trans[ "generation_time" ] == 1 ] ) > 0, True, "nobody is infected by someone who is infected by for one unit of time" )

        # check the maximum time people are stil infections
        max_sd   = 7;
        max_time = float( params.get_param( "mean_infectious_period" ) ) + max_sd * float(  params.get_param( "sd_infectious_period" ) )
        np.testing.assert_equal( max( df_trans[ "generation_time" ] ) < max_time, True, "someone is infectious at a time greater than mean + 7 * std. dev. of the infectious curve " )

        # check that some people are infected across all networks
        np.testing.assert_equal( sum( df_trans[ "occupation_network_source" ] == constant.HOUSEHOLD ) > 0, True, "no transmission on the household network" )
        np.testing.assert_equal( sum( df_trans[ "occupation_network_source" ] == constant.OCCUPATION )      > 0, True, "no transmission on the work network" )
        np.testing.assert_equal( sum( df_trans[ "occupation_network_source" ] == constant.RANDOM )    > 0, True, "no transmission on the random network" )

        # check hospitalised people are not transmitting on the work and household networks
        np.testing.assert_equal( sum( ( df_trans[ "occupation_network_source" ] == constant.HOUSEHOLD ) & ( df_trans[ "status_source" ] == constant.EVENT_TYPES.HOSPITALISED ) ), 0, "hospitalised people transmitting on the household network" )
        np.testing.assert_equal( sum( ( df_trans[ "occupation_network_source" ] == constant.OCCUPATION ) &      ( df_trans[ "status_source" ] == constant.EVENT_TYPES.HOSPITALISED ) ), 0, "hospitalised people transmitting on the work network" )    

        
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
        
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params = utils.set_homogeneous_random_network_only(params,n_connections,end_time)
        params.set_param( "infectious_rate", infectious_rate )
        params.set_param( "mean_infectious_period", mean_infectious_period )
        params.set_param( "sd_infectious_period", sd_infectious_period )
        params.set_param( "n_seed_infection", n_seed_infection ) 
        params.set_param( "n_total", n_total ) 
        params.set_param( "rng_seed", 2 ) 
        params.set_param( "random_interaction_distribution", 0 );
        params.set_param( "hospital_on", 0 );

        # Make mild and asymptomatic infections as infectious as normal ones:
        params.set_param("mild_infectious_factor", 1)
        params.set_param("asymptomatic_infectious_factor", 1)

        # Make recovery happen slowly enough that we explore the full
        # infectiousness(tau) function, assumed for the analytical calculation.
        params.set_param("mean_time_to_recover",
        mean_infectious_period + 2 * sd_infectious_period)
        params.set_param("mean_asymptomatic_to_recovery",
                         mean_infectious_period + 2 * sd_infectious_period)
        params.set_param("sd_time_to_recover", 0.5)
        params.set_param("sd_asymptomatic_to_recovery", 0.5)

        params.write_params(constant.TEST_DATA_FILE)     
                
        # Call the model using baseline parameters, pipe output to file, read output file
        file_output   = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)      
        df_output     = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
        df_ts         = df_output.loc[ :, ["time", "total_infected"]]

        # calculate the rate exponential rate of grwoth from the model
        ts_1  = df_ts[ df_ts[ "total_infected" ] > ( n_total * fraction_1 ) ].min() 
        ts_2  = df_ts[ df_ts[ "total_infected" ] > ( n_total * fraction_2 ) ].min() 
        slope = ( log( ts_2[ "total_infected"]) - log( ts_1[ "total_infected"]) ) / ( ts_2[ "time" ] - ts_1[ "time" ] )

        # this is an analytical approximation in the limit of large 
        # mean_infectious_rate, but works very well for most values
        # unless there is large amount infection the following day
        theta     = sd_infectious_period * sd_infectious_period / mean_infectious_period
        k         = mean_infectious_period / theta
        char_func = lambda x: exp(x) - infectious_rate / ( 1 + x * theta )**k
        slope_an  = optimize.brentq( char_func, - 0.99 / theta, 1  )

        np.testing.assert_allclose( slope, slope_an, rtol = tolerance, err_msg = "exponential growth deviates too far from analytic approximation")
    
    
    def test_monoton_relative_transmission(
            self,
            end_time,
            transmission_NETWORK,
            relative_transmission_values
        ):
        """
        Test that monotonic change (increase, decrease, or equal) in relative_transmission_NETWORK values
        leads to corresponding change (increase, decrease, or equal) in counts of transmissions in the NETWORK.
        
        """
        relative_transmissions = [ "relative_transmission_household", "relative_transmission_occupation", "relative_transmission_random" ]
        relative_transmission = relative_transmissions[transmission_NETWORK]
        
        # calculate the transmission proportions for the first entry in the relative_transmission_values
        rel_trans_value_current = relative_transmission_values[0]
        
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params.set_param( "end_time", end_time )
        params.set_param( "relative_transmission_household", 1 )
        params.set_param( "relative_transmission_occupation", 1 )
        params.set_param( "relative_transmission_random", 1 )
        params.set_param( relative_transmission , rel_trans_value_current )
        params.write_params(constant.TEST_DATA_FILE)     

        file_output   = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)     
        df_trans_current = pd.read_csv(constant.TEST_TRANSMISSION_FILE, 
            comment = "#", sep = ",", skipinitialspace = True )
        
        # calculating the first ratio
        len_household = len( df_trans_current[ df_trans_current[ "infector_network" ] == constant.HOUSEHOLD ] )
        len_work = len( df_trans_current[ df_trans_current[ "infector_network" ] == constant.OCCUPATION ] ) 
        len_random = len( df_trans_current[ df_trans_current[ "infector_network" ] == constant.RANDOM ] )
        lengths = [int(len_household), int(len_work), int(len_random)]
        all_trans_current = sum(lengths)
        ratio_current = float( df_trans_current[ df_trans_current[ "infector_network" ] == transmission_NETWORK ].shape[0] ) / float(all_trans_current) 
        
        # calculate the transmission proportion for the rest and compare with the current
        for relative_transmission_value in relative_transmission_values[1:]:
            params.set_param(relative_transmission , relative_transmission_value )
            params.write_params(constant.TEST_DATA_FILE)     
    
            file_output   = open(constant.TEST_OUTPUT_FILE, "w")
            completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)     
            df_trans      = pd.read_csv(constant.TEST_TRANSMISSION_FILE, 
                comment = "#", sep = ",", skipinitialspace = True )
            
            all_trans = len( df_trans[ df_trans[ "infector_network" ] == constant.HOUSEHOLD ] ) + \
                        len( df_trans[ df_trans[ "infector_network" ] == constant.OCCUPATION ] ) + \
                        len( df_trans[ df_trans[ "infector_network" ] == constant.RANDOM ] )
            ratio_new = float( df_trans[ df_trans[ "infector_network" ] == transmission_NETWORK ].shape[0] ) / float(all_trans)
    
            # check the proportion of the transmissions
            if relative_transmission_value > rel_trans_value_current:
                np.testing.assert_equal( ratio_new > ratio_current, True)
            elif relative_transmission_value < rel_trans_value_current:
                np.testing.assert_equal( ratio_new < ratio_current, True)
            elif relative_transmission_value == rel_trans_value_current:
                np.testing.assert_allclose( ratio_new, ratio_current, atol = 0.01)
            
            # refresh current values
            ratio_current = ratio_new
            rel_trans_value_current = relative_transmission_value
    
    
    def test_monoton_fraction_asymptomatic(
            self,
            end_time,
            fraction_asymptomatic_0_9,
            fraction_asymptomatic_10_19,
            fraction_asymptomatic_20_29,
            fraction_asymptomatic_30_39,
            fraction_asymptomatic_40_49,
            fraction_asymptomatic_50_59,
            fraction_asymptomatic_60_69,
            fraction_asymptomatic_70_79,
            fraction_asymptomatic_80
        ):
        """
        Test that monotonic change (increase, decrease, or equal) in fraction_asymptomatic values
        leads to corresponding change (decrease, increase, or equal) in the total infections.
        
        """
        
        # calculate the total infections for the first entry in the fraction_asymptomatic values
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params.set_param( "end_time", end_time )
        params.set_param( "fraction_asymptomatic_0_9", fraction_asymptomatic_0_9[0] )
        params.set_param( "fraction_asymptomatic_10_19", fraction_asymptomatic_10_19[0] )
        params.set_param( "fraction_asymptomatic_20_29", fraction_asymptomatic_20_29[0] )
        params.set_param( "fraction_asymptomatic_30_39", fraction_asymptomatic_30_39[0] )
        params.set_param( "fraction_asymptomatic_40_49", fraction_asymptomatic_40_49[0] )
        params.set_param( "fraction_asymptomatic_50_59", fraction_asymptomatic_50_59[0] )
        params.set_param( "fraction_asymptomatic_60_69", fraction_asymptomatic_60_69[0] )
        params.set_param( "fraction_asymptomatic_70_79", fraction_asymptomatic_70_79[0] )
        params.set_param( "fraction_asymptomatic_80", fraction_asymptomatic_80[0] )
        params.write_params(constant.TEST_DATA_FILE)

        file_output   = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)     
        df_output     = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        # calculate the sum of fraction_asymptomatic for different age groups
        fraction_asymptomatic_current = fraction_asymptomatic_0_9[0] + \
                                        fraction_asymptomatic_10_19[0] + \
                                        fraction_asymptomatic_20_29[0] + \
                                        fraction_asymptomatic_30_39[0] + \
                                        fraction_asymptomatic_40_49[0] + \
                                        fraction_asymptomatic_50_59[0] + \
                                        fraction_asymptomatic_60_69[0] + \
                                        fraction_asymptomatic_70_79[0] + \
                                        fraction_asymptomatic_80[0]
        total_infected_current = df_output[ "total_infected" ].iloc[-1]
        
        # calculate the total infections for the rest and compare with the current
        for idx in range(1, len(fraction_asymptomatic_0_9)):
            params.set_param( "fraction_asymptomatic_0_9", fraction_asymptomatic_0_9[idx] )
            params.set_param( "fraction_asymptomatic_10_19", fraction_asymptomatic_10_19[idx] )
            params.set_param( "fraction_asymptomatic_20_29", fraction_asymptomatic_20_29[idx] )
            params.set_param( "fraction_asymptomatic_30_39", fraction_asymptomatic_30_39[idx] )
            params.set_param( "fraction_asymptomatic_40_49", fraction_asymptomatic_40_49[idx] )
            params.set_param( "fraction_asymptomatic_50_59", fraction_asymptomatic_50_59[idx] )
            params.set_param( "fraction_asymptomatic_60_69", fraction_asymptomatic_60_69[idx] )
            params.set_param( "fraction_asymptomatic_70_79", fraction_asymptomatic_70_79[idx] )
            params.set_param( "fraction_asymptomatic_80", fraction_asymptomatic_80[idx] )
            params.write_params(constant.TEST_DATA_FILE)     
    
            file_output   = open(constant.TEST_OUTPUT_FILE, "w")
            completed_run = subprocess.run([constant.command], 
                stdout = file_output, shell = True)     
            df_output_new     = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
            
            fraction_asymptomatic_new = fraction_asymptomatic_0_9[idx] + \
                                        fraction_asymptomatic_10_19[idx] + \
                                        fraction_asymptomatic_20_29[idx] + \
                                        fraction_asymptomatic_30_39[idx] + \
                                        fraction_asymptomatic_40_49[idx] + \
                                        fraction_asymptomatic_50_59[idx] + \
                                        fraction_asymptomatic_60_69[idx] + \
                                        fraction_asymptomatic_70_79[idx] + \
                                        fraction_asymptomatic_80[idx]
            total_infected_new = df_output_new[ "total_infected" ].iloc[-1]
    
            # check the total infections
            if fraction_asymptomatic_new > fraction_asymptomatic_current:
                np.testing.assert_equal( total_infected_new < total_infected_current, True)
            elif fraction_asymptomatic_new < fraction_asymptomatic_current:
                np.testing.assert_equal( total_infected_new > total_infected_current, True)
            elif fraction_asymptomatic_new == fraction_asymptomatic_current:
                np.testing.assert_allclose( total_infected_new, total_infected_current, atol = 0.01)
            
            # refresh current values
            fraction_asymptomatic_current = fraction_asymptomatic_new
            total_infected_current = total_infected_new
        
        
    def test_monoton_asymptomatic_infectious_factor(
            self,
            end_time,
            asymptomatic_infectious_factor
        ):
        """
        Test that monotonic change (increase, decrease, or equal) in asymptomatic_infectious_factor values
        leads to corresponding change (increase, decrease, or equal) in the total infections.
        
        """
        
        # calculate the total infections for the first entry in the asymptomatic_infectious_factor values
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params.set_param( "end_time", end_time )
        params.set_param( "asymptomatic_infectious_factor", asymptomatic_infectious_factor[0] )
        params.write_params(constant.TEST_DATA_FILE)     

        file_output   = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)     
        df_output     = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        # save the current asymptomatic_infectious_factor value
        asymptomatic_infectious_factor_current = asymptomatic_infectious_factor[0]
        total_infected_current = df_output[ "total_infected" ].iloc[-1]
        
        # calculate the total infections for the rest and compare with the current
        for idx in range(1, len(asymptomatic_infectious_factor)):
            params.set_param("asymptomatic_infectious_factor", asymptomatic_infectious_factor[idx])
            params.write_params(constant.TEST_DATA_FILE)
    
            file_output   = open(constant.TEST_OUTPUT_FILE, "w")
            completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
            df_output_new     = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
            
            asymptomatic_infectious_factor_new = asymptomatic_infectious_factor[idx]
            total_infected_new = df_output_new[ "total_infected" ].iloc[-1]
    
            # check the total infections
            if asymptomatic_infectious_factor_new > asymptomatic_infectious_factor_current:
                np.testing.assert_equal( total_infected_new > total_infected_current, True)
            elif asymptomatic_infectious_factor_new < asymptomatic_infectious_factor_current:
                np.testing.assert_equal( total_infected_new < total_infected_current, True)
            elif asymptomatic_infectious_factor_new == asymptomatic_infectious_factor_current:
                np.testing.assert_allclose( total_infected_new, total_infected_current, atol = 0.01)
            
            # refresh current values
            asymptomatic_infectious_factor_current = asymptomatic_infectious_factor_new
            total_infected_current = total_infected_new
            
            
            
    def test_monoton_relative_susceptibility(
            self,
            end_time,
            relative_susceptibility_0_9,
            relative_susceptibility_10_19,
            relative_susceptibility_20_29,
            relative_susceptibility_30_39,
            relative_susceptibility_40_49,
            relative_susceptibility_50_59,
            relative_susceptibility_60_69,
            relative_susceptibility_70_79,
            relative_susceptibility_80
        ):
        """
        Test that monotonic change (increase or decrease) in relative_susceptibility 
        leads to corresponding changes (increase or decrease) in the proportion of 
        the infections within each age group.
        
        """
        tolerance = 0.00001
        # set the first parameters
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params.set_param( "end_time", end_time )
        params.set_param( "relative_susceptibility_0_9", relative_susceptibility_0_9[0] )
        params.set_param( "relative_susceptibility_10_19", relative_susceptibility_10_19[0] )
        params.set_param( "relative_susceptibility_20_29", relative_susceptibility_20_29[0] )
        params.set_param( "relative_susceptibility_30_39", relative_susceptibility_30_39[0] )
        params.set_param( "relative_susceptibility_40_49", relative_susceptibility_40_49[0] )
        params.set_param( "relative_susceptibility_50_59", relative_susceptibility_50_59[0] )
        params.set_param( "relative_susceptibility_60_69", relative_susceptibility_60_69[0] )
        params.set_param( "relative_susceptibility_70_79", relative_susceptibility_70_79[0] )
        params.set_param( "relative_susceptibility_80", relative_susceptibility_80[0] )
        params.write_params(constant.TEST_DATA_FILE)     

        # get the current output
        file_output   = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)     
        df_trans_current = pd.read_csv(constant.TEST_TRANSMISSION_FILE, comment = "#", sep = ",", skipinitialspace = True )
        
        # calculate the proportion of infections in each age group
        inf_cur_zeros = pd.DataFrame([0]*9, columns=['zeros'], index=range(9))
        infected_current = df_trans_current['age_group_recipient'].value_counts().sort_index(axis=0)
        inf_cur_zeros = pd.concat([inf_cur_zeros,infected_current], ignore_index=True, axis=1)
        infected_current = inf_cur_zeros[0]+inf_cur_zeros.fillna(0)[1]
                                               
        # get the relative_susceptibility for all age groups
        relative_susceptibility_current = [ relative_susceptibility_0_9[0],
                                            relative_susceptibility_10_19[0],
                                            relative_susceptibility_20_29[0],
                                            relative_susceptibility_30_39[0],
                                            relative_susceptibility_40_49[0],
                                            relative_susceptibility_50_59[0],
                                            relative_susceptibility_60_69[0],
                                            relative_susceptibility_70_79[0],
                                            relative_susceptibility_80[0] ]        
        
        # calculate the infections for the rest and compare with the current
        for idx in range(1, len(relative_susceptibility_0_9)):
            params.set_param( "relative_susceptibility_0_9", relative_susceptibility_0_9[idx] )
            params.set_param( "relative_susceptibility_10_19", relative_susceptibility_10_19[idx] )
            params.set_param( "relative_susceptibility_20_29", relative_susceptibility_20_29[idx] )
            params.set_param( "relative_susceptibility_30_39", relative_susceptibility_30_39[idx] )
            params.set_param( "relative_susceptibility_40_49", relative_susceptibility_40_49[idx] )
            params.set_param( "relative_susceptibility_50_59", relative_susceptibility_50_59[idx] )
            params.set_param( "relative_susceptibility_60_69", relative_susceptibility_60_69[idx] )
            params.set_param( "relative_susceptibility_70_79", relative_susceptibility_70_79[idx] )
            params.set_param( "relative_susceptibility_80", relative_susceptibility_80[idx] )
            params.write_params(constant.TEST_DATA_FILE)     
    
            file_output   = open(constant.TEST_OUTPUT_FILE, "w")
            completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
            df_trans_new = pd.read_csv(constant.TEST_TRANSMISSION_FILE, comment = "#", sep = ",", skipinitialspace = True )
            
            relative_susceptibility_new = [ relative_susceptibility_0_9[idx],
                                            relative_susceptibility_10_19[idx],
                                            relative_susceptibility_20_29[idx],
                                            relative_susceptibility_30_39[idx],
                                            relative_susceptibility_40_49[idx],
                                            relative_susceptibility_50_59[idx],
                                            relative_susceptibility_60_69[idx],
                                            relative_susceptibility_70_79[idx],
                                            relative_susceptibility_80[idx] ]
            # calculate new number of infecteds in each age group
            inf_new_zeros = pd.DataFrame([0]*9, columns=['zeros'], index=range(9))
            infected_new = df_trans_new['age_group_recipient'].value_counts().sort_index(axis = 0)
            inf_new_zeros = pd.concat([inf_new_zeros,infected_new], ignore_index=True, axis=1)
            infected_new = inf_new_zeros[0]+inf_new_zeros.fillna(0)[1]
            
            # detect the age group whose current and new parameters values do not match
            nonmatch_pairs = np.array( [ [int(i), curr, newr] for i, (curr, newr) in enumerate(zip(relative_susceptibility_current, relative_susceptibility_new)) if curr != newr] )
            
            # for that age group
            if len(nonmatch_pairs) > 0:
                ids = nonmatch_pairs[:,0].tolist()
                shortlist_current = nonmatch_pairs[:,1].tolist()
                shortlist_new = nonmatch_pairs[:,2].tolist()
            
                # conduct the monotonicity check on actual Numbers of infections 
                for j, age in enumerate(ids):
                    if shortlist_current[j] - shortlist_new[j] > tolerance:
                        np.testing.assert_equal( infected_current[int(age)] > infected_new[int(age)], True)
                    if shortlist_new[j] - shortlist_current[j]  > tolerance:
                        np.testing.assert_equal( infected_new[int(age)] > infected_current[int(age)], True)
                    if abs(shortlist_new[j] - shortlist_current[j]) < tolerance:
                        np.testing.assert_allclose( infected_new[int(age)], infected_current[int(age)], atol = tolerance)
                            
            # refresh current values
            relative_susceptibility_current = relative_susceptibility_new
            infected_current = infected_new.copy()


    def test_monoton_mild_infectious_factor(
            self,
            end_time,
            mild_fraction_0_9,
            mild_fraction_10_19,
            mild_fraction_20_29,
            mild_fraction_30_39,
            mild_fraction_40_49,
            mild_fraction_50_59,
            mild_fraction_60_69,
            mild_fraction_70_79,
            mild_fraction_80,
            mild_infectious_factor
        ):
        """
        Test that monotonic change (increase, decrease, or equal) in mild_infectious_factor values
        leads to corresponding change (increase, decrease, or equal) in the total infections.
        
        """
        
        # calculate the total infections for the first entry in the asymptomatic_infectious_factor values
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params.set_param( "end_time", end_time )
        params.set_param( "mild_fraction_0_9", mild_fraction_0_9 )
        params.set_param( "mild_fraction_10_19", mild_fraction_10_19 )
        params.set_param( "mild_fraction_20_29", mild_fraction_20_29 )
        params.set_param( "mild_fraction_30_39", mild_fraction_30_39 )
        params.set_param( "mild_fraction_40_49", mild_fraction_40_49 )
        params.set_param( "mild_fraction_50_59", mild_fraction_50_59 )
        params.set_param( "mild_fraction_60_69", mild_fraction_60_69 )
        params.set_param( "mild_fraction_70_79", mild_fraction_70_79 )
        params.set_param( "mild_fraction_80", mild_fraction_80 )
        params.set_param( "mild_infectious_factor", mild_infectious_factor[0] )
        params.write_params(constant.TEST_DATA_FILE)     

        file_output   = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)     
        df_output     = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        # save the current mild_infectious_factor value
        mild_infectious_factor_current = mild_infectious_factor[0]
        total_infected_current = df_output[ "total_infected" ].iloc[-1]
        
        # calculate the total infections for the rest and compare with the current
        for idx in range(1, len(mild_infectious_factor)):
            params.set_param( "end_time", end_time )
            params.set_param( "mild_fraction_0_9", mild_fraction_0_9 )
            params.set_param( "mild_fraction_10_19", mild_fraction_10_19 )
            params.set_param( "mild_fraction_20_29", mild_fraction_20_29 )
            params.set_param( "mild_fraction_30_39", mild_fraction_30_39 )
            params.set_param( "mild_fraction_40_49", mild_fraction_40_49 )
            params.set_param( "mild_fraction_50_59", mild_fraction_50_59 )
            params.set_param( "mild_fraction_60_69", mild_fraction_60_69 )
            params.set_param( "mild_fraction_70_79", mild_fraction_70_79 )
            params.set_param( "mild_fraction_80", mild_fraction_80 )
            params.set_param("mild_infectious_factor", mild_infectious_factor[idx])
            params.write_params(constant.TEST_DATA_FILE)
    
            file_output   = open(constant.TEST_OUTPUT_FILE, "w")
            completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
            df_output_new     = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
            
            mild_infectious_factor_new = mild_infectious_factor[idx]
            total_infected_new = df_output_new[ "total_infected" ].iloc[-1]
    
            # check the total infections
            if mild_infectious_factor_new > mild_infectious_factor_current:
                np.testing.assert_equal( total_infected_new > total_infected_current, True)
            elif mild_infectious_factor_new < mild_infectious_factor_current:
                np.testing.assert_equal( total_infected_new < total_infected_current, True)
            elif mild_infectious_factor_new == mild_infectious_factor_current:
                np.testing.assert_allclose( total_infected_new, total_infected_current, atol = 0.01)
            
            # refresh current values
            mild_infectious_factor_current = mild_infectious_factor_new
            total_infected_current = total_infected_new


    def test_ratio_presymptomatic_symptomatic( 
            self, 
            n_total,
            n_seed_infection,
            end_time
        ):
        """
        Test that ratio presymptomatic to symptomatic individuals is correct; currently must be 1.
        """
        tolerance = 1/n_total

        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)

        params.set_param("n_total", n_total)
        params.set_param("n_seed_infection", n_seed_infection)
        params.set_param("end_time", end_time)
        params.write_params(constant.TEST_DATA_FILE)
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout=file_output, shell=True)
        df_indiv = pd.read_csv(
            constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True
        )
        
        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_indiv = pd.merge(df_indiv, df_trans, 
            left_on = "ID", right_on = "ID_recipient", how = "left")

        # all presymptomatic vs symptomatic
        N_severe_pre = len(
            df_indiv[
                (df_indiv["time_infected"] > 0) & (df_indiv["time_presymptomatic_severe"] > 0)
            ]
        )
        N_mild_pre = len(
            df_indiv[
                (df_indiv["time_infected"] > 0) & (df_indiv["time_presymptomatic_mild"] > 0)
            ]
        )
        
        N_severe_sym = len(
            df_indiv[
                (df_indiv["time_infected"] > 0) & (df_indiv["time_symptomatic_severe"] > 0)
            ]
        )
        N_mild_sym = len(
            df_indiv[
                (df_indiv["time_infected"] > 0) & (df_indiv["time_symptomatic_mild"] > 0)
            ]
        )
        N_pre = N_severe_pre + N_mild_pre
        N_sym = N_severe_sym + N_mild_sym
        
        np.testing.assert_allclose(
            N_pre, N_sym, atol=tolerance
        )

        # presymptomatic vs symptomatic by age
        for idx in range( constant.N_AGE_GROUPS ):
            N_severe_pre = len(
            df_indiv[
                (df_indiv["time_infected"] > 0) & (df_indiv["time_presymptomatic_severe"] > 0)
                & (df_indiv["age_group"] == constant.AGES[idx])
                ]
            )
            N_mild_pre = len(
                df_indiv[
                    (df_indiv["time_infected"] > 0) & (df_indiv["time_presymptomatic_mild"] > 0)
                    & (df_indiv["age_group"] == constant.AGES[idx])
                ]
            )
            
            N_severe_sym = len(
                df_indiv[
                    (df_indiv["time_infected"] > 0) & (df_indiv["time_symptomatic_severe"] > 0)
                    & (df_indiv["age_group"] == constant.AGES[idx])
                ]
            )
            N_mild_sym = len(
                df_indiv[
                    (df_indiv["time_infected"] > 0) & (df_indiv["time_symptomatic_mild"] > 0)
                    & (df_indiv["age_group"] == constant.AGES[idx])
                ]
            )
            N_pre = N_severe_pre + N_mild_pre
            N_sym = N_severe_sym + N_mild_sym
            
            np.testing.assert_allclose(
                N_pre, N_sym, atol = tolerance
            )
            
        
    def test_relative_transmission_update(self, test_params, update_relative_transmission_household, update_relative_transmission_occupation, update_relative_transmission_random ):
        """
           Check to that if we change the relative transmission parameters after day 1 we get 
           the same result as if we started with the same values
        """
        tol      = 0.05
        max_time = test_params["end_time"]+1;

        # run the baseline parameters
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model  = utils.get_model_swig( params )
        
        for time in range(max_time):
            model.one_time_step()
        model.write_transmissions()
        df_base = pd.read_csv( constant.TEST_TRANSMISSION_FILE, comment="#", sep=",", skipinitialspace=True )        
        df_base = df_base[ df_base["time_infected"] == max_time ]
        df_base = df_base.groupby(["infector_network"]).size().reset_index(name="n_infections") 
        
        del model
        del params
            
        # run the baseline parameters then at base-line update one
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model  = utils.get_model_swig( params )
        
        for time in range(test_params["end_time"]):
            model.one_time_step()
        model.update_running_params( "relative_transmission_household",  update_relative_transmission_household )
        model.update_running_params( "relative_transmission_occupation", update_relative_transmission_occupation )
        model.update_running_params( "relative_transmission_random",     update_relative_transmission_random )
        model.one_time_step()   
        model.write_transmissions()
        df_update = pd.read_csv( constant.TEST_TRANSMISSION_FILE, comment="#", sep=",", skipinitialspace=True )  
        df_update = df_update[ df_update["time_infected"] == max_time ]
        df_update = df_update.groupby(["infector_network"]).size().reset_index(name="n_infections") 
        
        # check the change in values is in tolerance - wide tolerance bands due to saturation effects
        base     = df_base.loc[0,{"n_infections"}]["n_infections"] 
        expected = base * update_relative_transmission_household / test_params["relative_transmission_household"]
        actual   = df_update.loc[0,{"n_infections"}]["n_infections"]
        np.testing.assert_allclose(actual, expected, atol = base * tol, err_msg = "Number of transmissions did not change by expected amount after updating parameter")
                
        base     = df_base.loc[1,{"n_infections"}]["n_infections"] 
        expected = base * update_relative_transmission_occupation / test_params["relative_transmission_occupation"]
        actual   = df_update.loc[1,{"n_infections"}]["n_infections"]
        np.testing.assert_allclose(actual, expected, atol = base * tol, err_msg = "Number of transmissions did not change by expected amount after updating parameter")
        
        base     = df_base.loc[2,{"n_infections"}]["n_infections"] 
        expected = base * update_relative_transmission_random / test_params["relative_transmission_random"]
        actual   = df_update.loc[2,{"n_infections"}]["n_infections"]
        np.testing.assert_allclose(actual, expected, atol = base * tol, err_msg = "Number of transmissions did not change by expected amount after updating parameter")
       
    
         
    def test_single_seed_infections( self, test_params ):
        """
           Check that each person is only infected once in the seed infections
        """
     
        # run the baseline parameters
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model  = utils.get_model_swig( params )

        model.one_time_step()
        model.write_transmissions()
        df_trans = pd.read_csv( constant.TEST_TRANSMISSION_FILE, comment="#", sep=",", skipinitialspace=True )  
        df_trans[ "n_inf_type" ] = (df_trans[ "time_asymptomatic" ] == 0 )*1 + ( df_trans[ "time_presymptomatic_mild" ] == 0 )*1 + ( df_trans[ "time_presymptomatic_severe" ] == 0 )*1
                                   
        np.testing.assert_equal( len( df_trans ), test_params["n_seed_infection"], "The number of seed infections is not equal to the size of the transmission file")
        np.testing.assert_equal( sum( df_trans["n_inf_type"] >1 ), 0, "Individuals with more than one type of infections" )
        np.testing.assert_equal( sum( df_trans["n_inf_type"] == 1), test_params["n_seed_infection"], "Number of transmission with more than one type is not equal to the number of seed infections" )
        
        
        
    def test_presymptomatic_symptomatic_transmissions( 
            self, 
            n_total,
            n_seed_infection,
            end_time
        ):
        """
        Test that presymptomatic and symptomatic individuals transmit as expected
        """
        tolerance = 0.05        
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("self_quarantine_fraction", 0)

        params.set_param("hospitalised_fraction_0_9", 0)
        params.set_param("hospitalised_fraction_10_19", 0)
        params.set_param("hospitalised_fraction_20_29", 0)
        params.set_param("hospitalised_fraction_30_39", 0)
        params.set_param("hospitalised_fraction_40_49", 0)
        params.set_param("hospitalised_fraction_50_59", 0)
        params.set_param("hospitalised_fraction_60_69", 0)
        params.set_param("hospitalised_fraction_70_79", 0)
        params.set_param("hospitalised_fraction_80", 0)
            
       
        params.set_param("mean_infectious_period", 6)
        params.set_param("sd_infectious_period", 2.5)
        params.set_param("mean_time_to_recover", 60)
        params.set_param("sd_time_to_recover", 1)
        params.set_param("mean_time_to_hospital", 60)
        params.set_param("mean_time_to_symptoms", 6)
        params.set_param("sd_time_to_symptoms", 2.5)
        
        params.set_param("relative_transmission_household", 0)
        params.set_param("relative_transmission_occupation", 0)
        
        params.set_param("mean_work_interactions_child", 0)
        params.set_param("mean_work_interactions_adult", 0)
        params.set_param("mean_work_interactions_elderly", 0)
        params.set_param("daily_fraction_work", 0)
        
        params.set_param("n_total", n_total)
        params.set_param("n_seed_infection", n_seed_infection)
        params.set_param("end_time", end_time)
        
        params.write_params(constant.TEST_DATA_FILE)
        
        file_output   = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)     
        df_output     = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
        df_trans      = pd.read_csv(constant.TEST_TRANSMISSION_FILE, comment = "#", sep = ",", skipinitialspace = True )
 
        # check to see that the number of entries in the transmission file is that in the time-series
        np.testing.assert_equal( len( df_trans ), df_output.loc[ :, "total_infected" ].max(), "length of transmission file is not the number of infected in the time-series" )
        
        # check if hospitalised and ICU-ed people infect anybody
        np.testing.assert_equal( len( df_trans[ df_trans[ "status_source" ] == constant.EVENT_TYPES.HOSPITALISED.value] ) , 0," transmission from hospitalised people" )
        np.testing.assert_equal( len( df_trans[ df_trans[ "status_source" ] == constant.EVENT_TYPES.CRITICAL.value] )     , 0," transmission from critical people" )
        
        # Find the time at which a small fraction of the population has been
        # infected, then restrict to all transmission events where the infector
        # was infected before then. It needs to be small enough that people
        # infected just before this point can get through their whole infectious
        # period without encountering epidemic saturation, but large enough to
        # give minimal noise. 
        # by Chris Wymant 
        # <<<     
        fraction_mid_expo_phase = 0.001
        df_output  = df_output[ df_output[ "total_infected" ] < ( n_total * fraction_mid_expo_phase ) ].max()
        time_mid_expo_growth = df_output["time"]
        df_trans = df_trans[df_trans["time_infected_source"] < int(time_mid_expo_growth)]
        # >>> 
        
        N_presymptomatics = len( df_trans[ df_trans[ "status_source" ] == constant.EVENT_TYPES.PRESYMPTOMATIC.value] ) 
        N_symptomatics = len( df_trans[ df_trans[ "status_source" ] == constant.EVENT_TYPES.SYMPTOMATIC.value] ) 
        
#        np.testing.assert_equal( N_presymptomatics, N_symptomatics, "presymptomatic and symptomatic are not equally transmitting" )
     
        N_presymptomatics_mild = len( df_trans[ df_trans[ "status_source" ] == constant.EVENT_TYPES.PRESYMPTOMATIC_MILD.value] )
        N_symptomatics_mild = len( df_trans[ df_trans[ "status_source" ] == constant.EVENT_TYPES.SYMPTOMATIC_MILD.value] )
        N_involved = N_presymptomatics_mild+N_presymptomatics+N_symptomatics_mild+N_symptomatics
        
        np.testing.assert_allclose( (N_presymptomatics_mild+N_presymptomatics), N_involved*0.5, atol = N_involved*tolerance) 
        np.testing.assert_allclose( (N_symptomatics_mild+N_symptomatics), N_involved*0.5, atol = N_involved*tolerance) 

    

    def test_infectiousness_multiplier( self, test_params, sd_multipliers ):
        """
           Check that the total infected stays the same up to 0.5 SD.
        """
     
        ordered_multipliers = sorted( sd_multipliers )
        transmissions = []
        total_infected = []
        for sd_multiplier in ordered_multipliers:
          params = utils.get_params_swig()
          for param, value in test_params.items():
              params.set_param( param, value )  
          params.set_param( "sd_infectiousness_multiplier", sd_multiplier )
          model  = utils.get_model_swig( params )

          for time in range( test_params[ "end_time" ] ):
              model.one_time_step()

          results = model.one_time_step_results()
          total_infected.append( results[ "total_infected" ] )

          del model
          del params

        base_infected = total_infected[0]

        np.testing.assert_allclose([total_infected[0]]*len(total_infected), total_infected, rtol=0.05)

    

    def test_infectiousness_multiplier_transmissions_increase_with_multiplier( self, test_params, n_bins ):
        """
           Check that the mean number of infected increases across infectiousness bins.
        """
     
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model  = utils.get_model_swig( params )

        for time in range(test_params["end_time"]):
            model.one_time_step()

        model.write_transmissions()
        model.write_individual_file()

        df_trans = pd.read_csv( constant.TEST_TRANSMISSION_FILE, comment="#", sep=",", skipinitialspace=True )  
        df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )

        df_indiv.rename( columns = { "ID":"ID_source"}, inplace = True )

        df_indiv["im_bin"] = pd.qcut(df_indiv["infectiousness_multiplier"], n_bins, labels=False)

        source_trans = pd.merge( df_indiv, df_trans, on = "ID_source" )

        avg_trans = source_trans.groupby( [ "im_bin" ] ).size() / df_indiv.groupby( [ "im_bin" ] ).size()
        
        is_trans_cnt_increasing = avg_trans.diff()[1:] > 0

        np.testing.assert_equal( np.all(is_trans_cnt_increasing), True, "Infectiousness does not increase with multiplier" )
