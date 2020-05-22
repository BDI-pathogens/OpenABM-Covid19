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
        "test_network_connections_2way": [ 
            dict( n_total = 10000 ),
            dict( n_total = 20000 ),
            dict( n_total = 50000 ),
            dict( n_total = 100000 ),
            dict( n_total = 250000 )
        ], 
        "test_household_network": [ 
            dict( n_total = 10000 ),
            dict( n_total = 20000 ),
            dict( n_total = 30000 ),
            dict( n_total = 50000 ),
            dict( n_total = 100000 )
        ],
        "test_random_network": [ 
            dict( 
                mean_random_interactions_child   = 2,
                sd_random_interactions_child     = 2,
                mean_random_interactions_adult   = 4,
                sd_random_interactions_adult     = 4,
                mean_random_interactions_elderly = 3,
                sd_random_interactions_elderly   = 3,
                n_total                          = 50000
            ),
            dict( 
                mean_random_interactions_child   = 1,
                sd_random_interactions_child     = 2,
                mean_random_interactions_adult   = 1,
                sd_random_interactions_adult     = 2,
                mean_random_interactions_elderly = 1,
                sd_random_interactions_elderly   = 2,
                n_total                          = 100000
            ),
               dict( 
                mean_random_interactions_child   = 0,
                sd_random_interactions_child     = 2,
                mean_random_interactions_adult   = 0,
                sd_random_interactions_adult     = 2,
                mean_random_interactions_elderly = 0,
                sd_random_interactions_elderly   = 2,
                n_total                          = 40000
            ),
            dict( 
                mean_random_interactions_child   = 4,
                sd_random_interactions_child     = 4,
                mean_random_interactions_adult   = 3,
                sd_random_interactions_adult     = 3,
                mean_random_interactions_elderly = 2,
                sd_random_interactions_elderly   = 2,
                n_total                          = 40000
            ),
            dict( 
                mean_random_interactions_child   = 8,
                sd_random_interactions_child     = 8,
                mean_random_interactions_adult   = 7,
                sd_random_interactions_adult     = 7,
                mean_random_interactions_elderly = 6,
                sd_random_interactions_elderly   = 6,
                n_total                          = 10000

            ),
        ],
        "test_occupation_network": [ 
            dict( 
                n_total = 40000,
                mean_work_interactions_child   = 10,
                mean_work_interactions_adult   = 7,
                mean_work_interactions_elderly = 3,
                daily_fraction_work            = 0.5
            ),
            dict( 
                n_total = 20000,
                mean_work_interactions_child   = 6,
                mean_work_interactions_adult   = 10,
                mean_work_interactions_elderly = 5,
                daily_fraction_work            = 0.43
            ),
            dict( 
                n_total = 20000,
                mean_work_interactions_child   = 4.63,
                mean_work_interactions_adult   = 3.25,
                mean_work_interactions_elderly = 5.85,
                daily_fraction_work            = 0.83
            ),
            dict( 
                n_total = 20000,
                mean_work_interactions_child   = 0,
                mean_work_interactions_adult   = 0,
                mean_work_interactions_elderly = 0,
                daily_fraction_work            = 0.5
            ),
            dict( 
                n_total = 20000,
                mean_work_interactions_child   = 2,
                mean_work_interactions_adult   = 6,
                mean_work_interactions_elderly = 10,
                daily_fraction_work            = 1.0
            )
        ],
        "test_occupation_network_proportions": [ 
            dict( 
                n_total = 40000,
                child_network_adults   = 0,
                elderly_network_adults = 0
            ),
            dict( 
                n_total = 40000,
                child_network_adults   = 0.01,
                elderly_network_adults = 0.01
            ),
            dict( 
                n_total = 40000,
                child_network_adults   = 0.1,
                elderly_network_adults = 0.1
            ),
            dict( 
                n_total = 40000,
                child_network_adults   = 0.2,
                elderly_network_adults = 0.2
            ),
            dict( 
                n_total = 40000,
                child_network_adults   = 0.45,
                elderly_network_adults = 0.45
            )
        ],
        "test_occupation_network_recurrence": [ 
            dict( 
                test_params = dict( 
                    n_total = 10000,
                    end_time = 15,
                    mean_work_interactions_child   = 10,
                    mean_work_interactions_adult   = 7,
                    mean_work_interactions_elderly = 3,
                    daily_fraction_work            = 0.5,
                    work_network_rewire            = 0.1
                )
            ),
            dict(
                test_params = dict( 
                    n_total = 10000,
                    end_time = 15,
                    mean_work_interactions_child   = 10,
                    mean_work_interactions_adult   = 7,
                    mean_work_interactions_elderly = 3,
                    daily_fraction_work            = 0.75,
                    work_network_rewire            = 0.2
                ),
            ),
            dict(
                test_params = dict( 
                    n_total = 10000,
                    end_time = 15,
                    mean_work_interactions_child   = 10,
                    mean_work_interactions_adult   = 7,
                    mean_work_interactions_elderly = 3,
                    daily_fraction_work            = 0.25,
                    work_network_rewire            = 0.3
                )
            )
        ]
    }
    """
    Test class for checking 
    """
    def test_network_connections_2way(self,n_total):
        """
        Test to check that for all connections in every persons interaction diary
        there is a corresponding connection in the other person's diary 
        """        
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params = utils.turn_off_interventions(params,1)  
        params.set_param("n_total",n_total)
        params.write_params(constant.TEST_DATA_FILE)        
      
        file_output   = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_int        = pd.read_csv(constant.TEST_INTERACTION_FILE, 
            comment = "#", sep = ",", skipinitialspace = True )
        
        left  = df_int.loc[ :, ['ID_1', 'ID_2'] ]        
        right = df_int.loc[ :, ['ID_1', 'ID_2'] ]
        right.rename( columns =  {"ID_1":"ID_2","ID_2":"ID_1"},inplace=True)

        left.drop_duplicates(keep="first",inplace=True)
        right.drop_duplicates(keep="first",inplace=True)
        join = pd.merge(left,right,on=["ID_1","ID_2"], how="inner")
        
        N_base = len( left )
        N_join = len( join )
        
        np.testing.assert_equal( N_base, N_join )
    
    def test_household_network(self,n_total):
        """
        Test to check that all interactions within a household are made
        """    
        
        # note when counting connections we count each end
        expectedConnections = pd.DataFrame(data={'size': [1,2,3,4,5,6], 'expected': [0,2,6,12,20,30]})
            
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params = utils.turn_off_interventions(params,1)
        params.set_param("n_total",n_total)
        params.write_params(constant.TEST_DATA_FILE)        

        file_output   = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
       
        # get the number of people in each house hold
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE, comment = "#", sep = ",", skipinitialspace = True )
        df_house = df_indiv.groupby(["house_no"]).size().reset_index(name="size")
     
        # get the number of interactions per person on the housegold n
        df_int  = pd.read_csv(constant.TEST_INTERACTION_FILE, comment = "#", sep = ",", skipinitialspace = True )
        df_int  = df_int[ df_int["type"] == constant.HOUSEHOLD]
        np.testing.assert_array_equal( df_int.loc[:,"house_no_1"], df_int.loc[:,"house_no_2"] )

        df_int  = df_int.groupby(["house_no_1"]).size().reset_index(name="connections")
        
        # see whether that is the expected number
        df_house = pd.merge( df_house, expectedConnections,on =[ "size" ], how = "left")
        df_house = pd.merge( df_house,df_int,
            left_on =[ "house_no" ], right_on = ["house_no_1"], how = "outer" )

        # check single person household without connections
        N_nocon        = df_house.loc[:,["connections"]].isnull().sum().sum()
        N_nocon_single = df_house[ df_house["size"] == 1].loc[ :,["connections"]].isnull().sum().sum()
        np.testing.assert_equal(  N_nocon, N_nocon_single )
        
        # check the rest are the same 
        df_house = df_house[ df_house["size"] > 1 ]
        np.testing.assert_array_equal( df_house.loc[:,"connections"], df_house.loc[:,"expected"] )
    
       
    def test_random_network( 
            self,
            mean_random_interactions_child,
            sd_random_interactions_child,
            mean_random_interactions_adult,
            sd_random_interactions_adult,
            mean_random_interactions_elderly,
            sd_random_interactions_elderly,
            n_total
        ):
        """
        Test to check that there are the correct number of interactions per
        individual on the random network
        """  
      
        # absoluta tolerance
        tolerance = 0.03
        
        # note when counting connections we count each end
        ageTypeMap = pd.DataFrame( data={
            "age_group": constant.AGES, 
            "age_type": constant.AGE_TYPES } )
        
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params = utils.turn_off_interventions(params,1)
        params.set_param("mean_random_interactions_child",  mean_random_interactions_child )
        params.set_param("sd_random_interactions_child",    sd_random_interactions_child )
        params.set_param("mean_random_interactions_adult",  mean_random_interactions_adult )
        params.set_param("sd_random_interactions_adult",    sd_random_interactions_adult )
        params.set_param("mean_random_interactions_elderly",mean_random_interactions_elderly )
        params.set_param("sd_random_interactions_elderly",  sd_random_interactions_elderly )
        params.set_param("n_total",n_total)
        params.write_params(constant.TEST_DATA_FILE)

        file_output   = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        
        # get all the people, need to hand case if people having zero connections
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE, comment = "#", sep = ",", skipinitialspace = True )
        df_indiv = df_indiv.loc[:,["ID","age_group"]] 
        df_indiv = pd.merge( df_indiv, ageTypeMap, on = "age_group", how = "left" )

        # get all the random connections
        df_int = pd.read_csv(constant.TEST_INTERACTION_FILE, comment = "#", sep = ",", skipinitialspace = True )
        df_int = df_int[ df_int["type"] == constant.RANDOM ]
        
        # check the correlation is below a threshold
        corr = df_int['house_no_1'].corr(df_int['house_no_2'])
        if ( len( df_int ) > 1 ) :
            np.testing.assert_allclose( corr, 0, atol = tolerance )
        
        df_int = df_int.loc[:,["ID_1"]] 
        df_int = df_int.groupby(["ID_1"]).size().reset_index(name="connections")
        df_int = pd.merge( df_indiv, df_int, left_on = "ID", right_on = "ID_1", how = "left" )
        df_int.fillna(0, inplace = True)
        
        # check mean and 
        mean = df_int[df_int["age_type"] == constant.CHILD].loc[:,"connections"].mean()
        sd   = df_int[df_int["age_type"] == constant.CHILD].loc[:,"connections"].std()    
        if mean_random_interactions_child > 0:
            np.testing.assert_allclose( sd,     sd_random_interactions_child, rtol = tolerance )
        
        mean = df_int[df_int["age_type"] == constant.ADULT].loc[:,"connections"].mean()
        sd   = df_int[df_int["age_type"] == constant.ADULT].loc[:,"connections"].std()        
        np.testing.assert_allclose( mean, mean_random_interactions_adult, rtol = tolerance )
        if mean_random_interactions_adult > 0:
            np.testing.assert_allclose( sd,   sd_random_interactions_adult, rtol = tolerance )
        
        mean = df_int[df_int["age_type"] == constant.ELDERLY].loc[:,"connections"].mean()
        sd   = df_int[df_int["age_type"] == constant.ELDERLY].loc[:,"connections"].std()        
        np.testing.assert_allclose( mean, mean_random_interactions_elderly, rtol = tolerance )
        if mean_random_interactions_elderly > 0:
            np.testing.assert_allclose( sd,   sd_random_interactions_elderly, rtol = tolerance )
  
    def test_occupation_network( 
            self,
            n_total,
            mean_work_interactions_child,
            mean_work_interactions_adult,
            mean_work_interactions_elderly,
            daily_fraction_work
        ):
        """
        Test to check that peoples work connections are on the correct network and
        that they have correct number on average
        """
        # absolute tolerance
        tolerance = 0.035

        primary_means = []
        secondary_means = []
        occupation_means = []
        retired_means = []
        elderly_means = []

        for i in range(1, 10):
            # absolute tolerance
            seed = i

            # note when counting connections we count each end
            ageTypeMap1 = pd.DataFrame(
                data={ "age_group_1": constant.AGES, "age_type_1": constant.AGE_TYPES } );
            ageTypeMap2 = pd.DataFrame(
                data={ "age_group_2": constant.AGES, "age_type_2": constant.AGE_TYPES } );

            paramByNetworkType = [
                mean_work_interactions_child,
                mean_work_interactions_adult,
                mean_work_interactions_elderly ]

            params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
            params = utils.turn_off_interventions(params,1)
            params.set_param("rng_seed", seed)
            params.set_param("n_total", n_total)
            params.set_param( "mean_work_interactions_child",   mean_work_interactions_child )
            params.set_param( "mean_work_interactions_adult",   mean_work_interactions_adult )
            params.set_param( "mean_work_interactions_elderly", mean_work_interactions_elderly )
            params.set_param( "daily_fraction_work",            daily_fraction_work )
            params.set_param( "n_total",n_total)
            params.write_params(constant.TEST_DATA_FILE)

            file_output   = open(constant.TEST_OUTPUT_FILE, "w")
            completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)

            # get all the people, need to hand case if people having zero connections
            df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE,
                comment = "#", sep = ",", skipinitialspace = True )
            df_indiv = df_indiv.loc[:,[ "ID", "age_group", "occupation_network" ] ]
            df_indiv = pd.merge( df_indiv, ageTypeMap1,
                left_on = "age_group", right_on = "age_group_1", how = "left" )

            # get all the work connections
            df_int  = pd.read_csv(constant.TEST_INTERACTION_FILE,
                comment = "#", sep = ",", skipinitialspace = True )
            df_int  = df_int[ df_int["type"] == constant.OCCUPATION ]
            df_int = pd.merge( df_int, ageTypeMap1,  on = "age_group_1", how = "left" )
            df_int = pd.merge( df_int, ageTypeMap2, on = "age_group_2", how = "left" )

            # get the number of connections for each person
            df_n_int = df_int.groupby( [ "ID_1" ] ).size().reset_index( name = "connections" )
            df_n_int = pd.merge( df_indiv, df_n_int, left_on = "ID", right_on = "ID_1", how = "left" )
            df_n_int.fillna( 0, inplace = True )

            # check there are connections for each age group
            for age in constant.AGES:
                if ( paramByNetworkType[ constant.NETWORK_TYPES[ constant.AGE_TYPES[ age ] ] ]  > 0 ) :
                    n = sum( df_int[ "age_group_1" ] == age )
                    np.testing.assert_equal( n > 0, True, "there are no work connections for age_group " + str( age ) )

            # check the correct people are on each network
            n = sum(
                ( df_int[ "age_group_1" ] == constant.AGE_0_9 ) &
                ( df_int[ "age_group_2" ] != constant.AGE_0_9 ) &
                ( df_int[ "age_type_2" ] != constant.ADULT )
                )
            np.testing.assert_equal( n, 0, "only 0_9 and adults on the 0_9 network" )

            n = sum(
                ( df_int[ "age_group_1" ] == constant.AGE_10_19 ) &
                ( df_int[ "age_group_2" ] != constant.AGE_10_19 ) &
                ( df_int[ "age_type_2" ] != constant.ADULT )
                )
            np.testing.assert_equal( n, 0, "only 10_19 and adults on the 10_19 network" )

            n = sum(
                ( df_int[ "age_group_1" ] == constant.AGE_70_79 ) &
                ( df_int[ "age_group_2" ] != constant.AGE_70_79 ) &
                ( df_int[ "age_type_2" ] != constant.ADULT )
                )
            np.testing.assert_equal( n, 0, "only 70_79 and adults on the 70_79 network" )

            n = sum(
                ( df_int[ "age_group_1" ] == constant.AGE_80 ) &
                ( df_int[ "age_group_2" ] != constant.AGE_80 ) &
                ( df_int[ "age_type_2" ] != constant.ADULT )
                )
            np.testing.assert_equal( n, 0, "only 80 adults on the 80 network" )

            # check the mean number of networks connections by network

            # for network in [ constant.RETIRED_NETWORK, constant.ELDERLY_NETWORK ]:
            primary_mean = df_n_int[ df_n_int[ "occupation_network" ] == constant.PRIMARY_NETWORK ].loc[:,"connections"].mean()
            primary_means.append(primary_mean)
            secondary_mean = df_n_int[ df_n_int[ "occupation_network" ] == constant.SECONDARY_NETWORK ].loc[:,"connections"].mean()
            secondary_means.append(secondary_mean)
                # np.testing.assert_allclose( mean, mean_work_interactions_child, rtol = tolerance )

            occupation_mean = df_n_int[ df_n_int[ "occupation_network" ] == constant.WORKING_NETWORK ].loc[:,"connections"].mean()
            occupation_means.append(occupation_mean)
            # np.testing.assert_allclose( mean, mean_work_interactions_adult, rtol = tolerance )

            # for network in [ constant.RETIRED_NETWORK, constant.ELDERLY_NETWORK ]:
            retired_mean = df_n_int[ df_n_int[ "occupation_network" ] == constant.RETIRED_NETWORK ].loc[:,"connections"].mean()
            retired_means.append(retired_mean)
            elderly_mean = df_n_int[ df_n_int[ "occupation_network" ] == constant.ELDERLY_NETWORK ].loc[:,"connections"].mean()
            elderly_means.append(elderly_mean)
                # np.testing.assert_allclose( mean, mean_work_interactions_elderly, rtol = tolerance )

            # check the correlation is below a threshold
            corr = df_int['house_no_1'].corr(df_int['house_no_2'])
            if ( len( df_int ) > 1 ) :
                np.testing.assert_allclose( corr, 0, atol = tolerance )

        df_primary_means = pd.DataFrame(primary_means)
        np.testing.assert_allclose( df_primary_means.mean(), mean_work_interactions_child, rtol = tolerance )

        df_secondary_means = pd.DataFrame(secondary_means)
        np.testing.assert_allclose( df_secondary_means.mean(), mean_work_interactions_child, rtol = tolerance )

        df_occupation_means = pd.DataFrame(occupation_means)
        np.testing.assert_allclose( df_occupation_means.mean(), mean_work_interactions_adult, rtol = tolerance )

        df_retired_means = pd.DataFrame(retired_means)
        np.testing.assert_allclose( df_retired_means.mean(), mean_work_interactions_elderly, rtol = tolerance )

        df_elderly_means = pd.DataFrame(elderly_means)
        np.testing.assert_allclose( df_elderly_means.mean(), mean_work_interactions_elderly, rtol = tolerance )


    def test_occupation_network_proportions( 
            self,
            n_total,
            child_network_adults,
            elderly_network_adults
        ):

        """
        Test to check proportions of adults in work networks
        """
      
        # absolute tolerance
        tolerance = 0.01

        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params = utils.turn_off_interventions(params,1)
        params.set_param("n_total",n_total)
        params.set_param( "child_network_adults",   child_network_adults )
        params.set_param( "elderly_network_adults",   elderly_network_adults )
        params.write_params(constant.TEST_DATA_FILE)        

        file_output   = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
       
        # get all the people, need to hand case if people having zero connections
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE, comment = "#", sep = ",", skipinitialspace = True )
        df_indiv = df_indiv.loc[:,[ "ID", "age_group", "occupation_network" ] ] 
        
        num_children_in_children = len( df_indiv[ 
                                                ( ( df_indiv["occupation_network"] == constant.PRIMARY_NETWORK) | ( df_indiv["occupation_network"] == constant.SECONDARY_NETWORK ) ) &
                                                ( ( df_indiv["age_group"] == constant.AGE_0_9 ) | ( df_indiv["age_group"] == constant.AGE_10_19 ) )
                                                ]
                                    )
        
        num_adults_in_children = len( df_indiv[
                                                ( ( df_indiv["occupation_network"] == constant.PRIMARY_NETWORK ) | ( df_indiv["occupation_network"] == constant.SECONDARY_NETWORK ) ) & 
                                                ( ( df_indiv["age_group"] == constant.AGE_20_29 ) | 
                                                  ( df_indiv["age_group"] == constant.AGE_30_39 ) | ( df_indiv["age_group"] == constant.AGE_40_49 ) | 
                                                  ( df_indiv["age_group"] == constant.AGE_50_59 ) | ( df_indiv["age_group"] == constant.AGE_60_69 ) )
                                              ]
                                    )
        
        num_elderly_in_elderly = len( df_indiv[ 
                                                ( ( df_indiv["occupation_network"] == constant.RETIRED_NETWORK ) | ( df_indiv["occupation_network"] == constant.ELDERLY_NETWORK ) ) &
                                                ( ( df_indiv["age_group"] == constant.AGE_70_79 ) | ( df_indiv["age_group"] == constant.AGE_80 ) )
                                                ]
                                    )
        
        num_adults_in_elderly = len( df_indiv[
                                                ( ( df_indiv["occupation_network"] == constant.RETIRED_NETWORK ) | ( df_indiv["occupation_network"] == constant.ELDERLY_NETWORK ) ) & 
                                                ( ( df_indiv["age_group"] == constant.AGE_20_29 ) | 
                                                  ( df_indiv["age_group"] == constant.AGE_30_39 ) | ( df_indiv["age_group"] == constant.AGE_40_49 ) | 
                                                  ( df_indiv["age_group"] == constant.AGE_50_59 ) | ( df_indiv["age_group"] == constant.AGE_60_69 ) )
                                              ]
                                    )
        
        num_adults_in_adults = len( df_indiv[
                                                ( df_indiv["occupation_network"] == constant.WORKING_NETWORK ) & 
                                                ( ( df_indiv["age_group"] == constant.AGE_20_29 ) | 
                                                  ( df_indiv["age_group"] == constant.AGE_30_39 ) | ( df_indiv["age_group"] == constant.AGE_40_49 ) | 
                                                  ( df_indiv["age_group"] == constant.AGE_50_59 ) | ( df_indiv["age_group"] == constant.AGE_60_69 ) )
                                              ]
                                    )

        num_adults_in_healthcare_worker_network = len(df_indiv[
                                                ( df_indiv["occupation_network"] == constant.HOSPITAL_WORK_NETWORK ) &
                                                ( ( df_indiv["age_group"] == constant.AGE_20_29) |
                                                  ( df_indiv["age_group"] == constant.AGE_30_39) | ( df_indiv["age_group"] == constant.AGE_40_49) |
                                                  ( df_indiv["age_group"] == constant.AGE_50_59) | ( df_indiv["age_group"] == constant.AGE_60_69) )
                                               ]
                                   )
        
        total_children_network = num_children_in_children + num_adults_in_children
        total_elderly_network = num_elderly_in_elderly + num_adults_in_elderly
        total = total_children_network + total_elderly_network + num_adults_in_adults + num_adults_in_healthcare_worker_network
        if ( total > 0 ):
            np.testing.assert_equal( total, n_total )
            
        if ( total_children_network > 0 ) :
            np.testing.assert_allclose( num_adults_in_children / num_children_in_children, child_network_adults, atol = tolerance )

        if ( total_elderly_network > 0 ) :
            np.testing.assert_allclose( num_adults_in_elderly / num_elderly_in_elderly, elderly_network_adults, atol = tolerance )
    
    def test_occupation_network_recurrence(self, test_params ):
        """
           Check to see that people only meet with the same person
           once per day on the occupational network
           
           Check to see that when you look over multiple days that 
           the mean number of unique contacts is mean_daily/daily_fraction
        """
        
        tol = 0.02
         
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model  = utils.get_model_swig( params )
           
        # step through time until we need to start to save the interactions each day
        model.one_time_step();   
        model.write_interactions_file()
        df_inter = pd.read_csv(constant.TEST_INTERACTION_FILE)
        df_inter[ "time" ] = 0

        for time in range( test_params[ "end_time" ] ):
            model.one_time_step();   
            model.write_interactions_file()
            df = pd.read_csv(constant.TEST_INTERACTION_FILE)
            df[ "time" ] = time + 1
            df_inter = df_inter.append( df )
  
        df_inter = df_inter[ df_inter[ "type" ] == constant.OCCUPATION ]
  
        # check to see there are sufficient daily connections and only one per set of contacts a day
        df_unique_daily = df_inter.groupby( ["time","ID_1","ID_2"]).size().reset_index(name="N");
        min_size = (test_params["end_time"]+1) * test_params[ "n_total"] *  min( 1,test_params["mean_work_interactions_child"],test_params["mean_work_interactions_adult"],test_params["mean_work_interactions_elderly"] )

        np.testing.assert_equal(sum(df_unique_daily["N"]==1)>min_size, True, "Less contacts than expected on the occuaptional networks" )
        np.testing.assert_equal(sum(df_unique_daily["N"]!=1), 0, "Repeat connections on same day on the occupational networks" )

        # check the mean unique connections over multiple days is mean/daily fraction
        df_unique = df_inter.groupby(["occupation_network_1","ID_1","ID_2"]).size().reset_index(name="N_unique")
        df_unique = df_unique.groupby(["occupation_network_1","ID_1"]).size().reset_index(name="N_conn")
        df_unique = df_unique.groupby(["occupation_network_1"]).mean()
    
        mean_by_type = [ test_params["mean_work_interactions_child"],test_params["mean_work_interactions_adult"],test_params["mean_work_interactions_elderly"]]
        
        for network in constant.NETWORKS:
            actual   = df_unique.loc[network,{"N_conn"}]["N_conn"]
            expected = mean_by_type[constant.NETWORK_TYPE_MAP[network]]/test_params["daily_fraction_work"]
            np.testing.assert_allclose(actual,expected,rtol=tol,err_msg="Expected mean unique occupational contacts over multiple days not as expected")
           
        
