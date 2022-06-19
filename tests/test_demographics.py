#!/usr/bin/env python3
"""
Created on Fri Mar 27 13:48:27 2020
Usage:
    pytest -rf tests/test_demographics.py::TestClass::test_demographic_proportions
    from the main folder
@author: anelnurtay
"""

import subprocess
import numpy as np, pandas as pd
import pytest
import random as rd

import sys
sys.path.append("src/COVID19")
from parameters import ParameterSet

from . import constant
from . import utilities as utils

def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )


class TestClass(object):
    params = {
        "test_demographic_proportions" : [
            dict( # UK
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
            dict( # even
                n_total     = 50000,
                population_0_9  = 50000 * 0.111,
                population_10_19= 50000 * 0.111,
                population_20_29= 50000 * 0.111,
                population_30_39= 50000 * 0.111,
                population_40_49= 50000 * 0.111,
                population_50_59= 50000 * 0.111,
                population_60_69= 50000 * 0.111,
                population_70_79= 50000 * 0.111,
                population_80= 50000 * 0.111
            ),
            dict( # Japan 2019
                n_total     = 100000,
                population_0_9  = 100000 * 0.08,
                population_10_19= 100000 * 0.09,
                population_20_29= 100000 * 0.10,
                population_30_39= 100000 * 0.12,
                population_40_49= 100000 * 0.15,
                population_50_59= 100000 * 0.13,
                population_60_69= 100000 * 0.13,
                population_70_79= 100000 * 0.13,
                population_80= 100000 * 0.09
            ),
#            dict(  # Nigeria 2019  FAILS 
#                   # "because the reference household panel does not include 
#                   # sufficient households with large numbers of children"
#                n_total     = 250000,
#                population_0_9  = 250000 * 0.312,
#                population_10_19= 250000 * 0.230,
#                population_20_29= 250000 * 0.160,
#                population_30_39= 250000 * 0.119,
#                population_40_49= 250000 * 0.082,
#                population_50_59= 250000 * 0.052,
#                population_60_69= 250000 * 0.030,
#                population_70_79= 250000 * 0.013,
#                population_80= 250000 * 0.002
#            ),
            dict( # Kazakhstan 2019  
                n_total     = 250000,
                population_0_9  = 250000 * 0.21,
                population_10_19= 250000 * 0.14,
                population_20_29= 250000 * 0.15,
                population_30_39= 250000 * 0.16,
                population_40_49= 250000 * 0.12,
                population_50_59= 250000 * 0.11,
                population_60_69= 250000 * 0.07,
                population_70_79= 250000 * 0.03,
                population_80= 250000 * 0.02
            ),
            dict( # UK
                n_total     = 500000,
                population_0_9  = 500000 * 0.12,
                population_10_19= 500000 * 0.11,
                population_20_29= 500000 * 0.13,
                population_30_39= 500000 * 0.13,
                population_40_49= 500000 * 0.13,
                population_50_59= 500000 * 0.13,
                population_60_69= 500000 * 0.11,
                population_70_79= 500000 * 0.08,
                population_80= 500000 * 0.05
            )
        ],
        "test_household_size" : [
            dict(
                n_total  = 10000, # default sizes
                household_size_1 = 0.29,
                household_size_2 = 0.34,
                household_size_3 = 0.15,
                household_size_4 = 0.13,
                household_size_5 = 0.04,
                household_size_6 = 0.02
            ),
            dict(
                n_total  = 10000, # shift from small to large
                household_size_1 = 0.24,
                household_size_2 = 0.29,
                household_size_3 = 0.10,
                household_size_4 = 0.18,
                household_size_5 = 0.09,
                household_size_6 = 0.07
            ),
            dict(
                n_total  =10000, # shift from large to small
                household_size_1 = 0.33,
                household_size_2 = 0.37,
                household_size_3 = 0.18,
                household_size_4 = 0.10,
                household_size_5 = 0.02,
                household_size_6 = 0.01
            ),
            dict(
                n_total  = 10000,# shift from medium
                household_size_1 = 0.32,
                household_size_2 = 0.37,
                household_size_3 = 0.08,
                household_size_4 = 0.05,
                household_size_5 = 0.08,
                household_size_6 = 0.06
            )
        ],
        "test_user_demographics": [ 
            dict( 
                test_params = dict(
                    n_total  = 1e3,
                    end_time = 20,
                ),
                n_houses = 1e1
            )
        ],
    }
    """
    Test class for checking 
    """
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
        
        error_tolerance = 0.01
        
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params.set_param("n_total", n_total)
        params.set_param("end_time", 1)
        params.set_param("population_0_9",population_0_9)
        params.set_param("population_10_19",population_10_19)
        params.set_param("population_20_29",population_20_29)
        params.set_param("population_30_39",population_30_39)
        params.set_param("population_40_49",population_40_49)   
        params.set_param("population_50_59",population_50_59)   
        params.set_param("population_60_69",population_60_69)
        params.set_param("population_70_79",population_70_79)
        params.set_param("population_80",population_80)
       
        population_fraction = [ population_0_9,   population_10_19, population_20_29,
                                  population_30_39, population_40_49, population_50_59,
                                  population_60_69, population_70_79, population_80 ]
       
        params.write_params(constant.TEST_DATA_FILE)        
        # file_output = open(constant.TEST_OUTPUT_FILE, "w")
        # completed_run = subprocess.run([constant.command], stdout=file_output,
        #     stderr=file_output, shell=True)
        # np.testing.assert_equal(completed_run.returncode, 0)

        # get the model and run for the required time steps
        mparams = utils.get_params_custom()
        model = utils.get_model_swig( mparams )
        for time in range( 1 ):
            model.one_time_step()
            
        # get the individual file and check age and houses
        model.write_individual_file()

        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE, 
            comment = "#", sep = ",", skipinitialspace = True )

        # population proportion by age
        N_tot = len( df_indiv )
        for idx in range( constant.N_AGE_GROUPS ):
           N = len(df_indiv[(df_indiv['age_group'] == constant.AGES[idx])])
           np.testing.assert_allclose(N, population_fraction[idx], atol=N_tot * error_tolerance)

    def test_household_size(self, n_total, household_size_1, household_size_2,
        household_size_3, household_size_4, household_size_5, household_size_6):
        """
        Test to check the household size distribution
        """

        # Set the parameters we want for the simulation.
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("end_time", 1)
        params.set_param("n_total", n_total)
        params.set_param("household_size_1", household_size_1)
        params.set_param("household_size_2", household_size_2)
        params.set_param("household_size_3", household_size_3)
        params.set_param("household_size_4", household_size_4)
        params.set_param("household_size_5", household_size_5)
        params.set_param("household_size_6", household_size_6)
        params.write_params(constant.TEST_DATA_FILE)

        # Calculate the number of people expected to be living in households of
        # each different size, based on the parameter definitions.
        household_size_counts = [household_size_1, household_size_2,
        household_size_3, household_size_4, household_size_5, household_size_6]
        household_size_counts_weighted = np.array(
        [count * (i + 1) for i, count in enumerate(household_size_counts)],
        dtype=float)
        household_size_counts_weighted *= float(n_total) / \
        sum(household_size_counts_weighted)

        # Run the simulation.
        # file_output = open(constant.TEST_OUTPUT_FILE, "w")
        # completed_run = subprocess.run([constant.command], stdout=file_output,
        #     stderr=file_output, shell=True)
        # np.testing.assert_equal(completed_run.returncode, 0)

        # get the model and run for the required time steps
        mparams = utils.get_params_custom()
        model = utils.get_model_swig( mparams )
        for time in range( 1 ):
            model.one_time_step()
            
        # get the individual file and check age and houses
        model.write_individual_file()

        # Find the number of people living in households of each different size
        # in the simulation output.
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE, 
            comment="#", sep=",", skipinitialspace = True)
        
        df_house = df_indiv.groupby(["house_no"]).size().reset_index(name="size")
        df_house = df_house.groupby(["size"]).size().reset_index(
        name="house_count")
        df_house["people_count"] = df_house["size"] * df_house["house_count"]
        df_house["people_count_expected"] = household_size_counts_weighted

        # Test!
        np.testing.assert_allclose(df_house["people_count"],
                                   df_house["people_count_expected"], rtol=0.02) 
        
        
    def test_user_demographics(self, test_params, n_houses):
        """
            Adds in a user defined demographic and household structure
        """
 
        n_houses     = int(n_houses)
        n_total      = int(test_params["n_total"])
        n_age_groups = int(constant.N_AGE_GROUPS)
         
        # build a random household/demographic structure
        IDs    = range(n_total)
        houses = [0]*n_total
        ages   = [0]*n_total
        for i in range(n_total):
            houses[i] = round( i / 4 )
#            houses[i] = rd.randrange(0,n_houses-1)
            ages[i]   = rd.randrange(0,n_age_groups-1)    
        houses = sorted( houses )
        df_demo = pd.DataFrame({'ID':np.array(IDs, dtype='int32'),'age_group':np.array(ages, dtype='int32'),'house_no':np.array(houses, dtype='int32')})
       
        # get the intial paramters and add user defined demographics 
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        params.set_demographic_household_table(df_demo)
        
        # get the model and run for the required time steps
        model = utils.get_model_swig( params )
        for time in range( test_params[ "end_time" ] ):
            model.one_time_step()
            
        # get the individual file and check age and houses
        model.write_individual_file()
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        
        # now check the households ages are correct
        df = pd.merge(df_demo,df_indiv,on = ["ID"], how = "left")
        n_wrong_age   = sum( df["age_group_x"]!=df["age_group_y"])
        n_wrong_house = sum( df["house_no_x"]!=df["house_no_y"])
        np.testing.assert_equal( n_wrong_age, 0, err_msg = "people in the wrong age group")
        np.testing.assert_equal( n_wrong_house, 0, err_msg = "people in the wrong house")
        
        
        
        
        
        
        
