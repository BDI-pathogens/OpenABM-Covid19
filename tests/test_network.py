#!/usr/bin/env python3
"""
Tests of the individual-based model, COVID19-IBM, using the individual file

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be 
run by calling 'pytest' from project folder.  

Created: March 2020
Author: p-robot
"""

import pytest, sys, shutil, os
import math
import numpy as np, pandas as pd
import random as rd
from scipy import optimize

sys.path.append("src/COVID19")
from parameters import ParameterSet
import COVID19.model as abm

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
                sd_random_interactions_child     = 1.5,
                mean_random_interactions_adult   = 1,
                sd_random_interactions_adult     = 1.5,
                mean_random_interactions_elderly = 1,
                sd_random_interactions_elderly   = 1.5,
                n_total                          = 50000
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
                n_total                          = 100000
            ),
            dict( 
                mean_random_interactions_child   = 8,
                sd_random_interactions_child     = 8,
                mean_random_interactions_adult   = 7,
                sd_random_interactions_adult     = 7,
                mean_random_interactions_elderly = 6,
                sd_random_interactions_elderly   = 6,
                n_total                          = 30000

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
        ],
        "test_user_defined_network": [
            dict( 
                test_params = dict(
                    n_total  = 1e4,
                    end_time = 20,
                    mean_time_to_critical = 30
                ),
                new_connections = 1e5
            )
        ],
        "test_custom_occupation_network": [
            dict(
                test_params = dict(
                    n_total = 10000,
                    end_time = 5,
                    mean_work_interactions_child   = 12,
                    mean_work_interactions_adult   = 5,
                    mean_work_interactions_elderly = 4,
                    daily_fraction_work            = 0.5,
                    work_network_rewire            = 0.1
                )
            ),
            dict(
                test_params = dict(
                    n_total = 10000,
                    end_time = 5,
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
                    mean_work_interactions_child   = 6,
                    mean_work_interactions_adult   = 12,
                    mean_work_interactions_elderly = 5,
                    daily_fraction_work            = 0.25,
                    work_network_rewire            = 0.3
                )
            ),
            dict(
                test_params = dict(
                    n_total = 20000,
                    end_time = 15,
                    mean_work_interactions_child   = 6,
                    mean_work_interactions_adult   = 12,
                    mean_work_interactions_elderly = 5,
                    daily_fraction_work            = 0.25,
                    work_network_rewire            = 0.3
                )
            ),
            dict(
                test_params = dict(
                    n_total = 6000,
                    end_time = 15,
                    mean_work_interactions_child   = 6,
                    mean_work_interactions_adult   = 12,
                    mean_work_interactions_elderly = 5,
                    daily_fraction_work            = 0.25,
                    work_network_rewire            = 0.3
                )
            ),
            dict(
                test_params = dict(
                    n_total = 2000,
                    end_time = 15,
                    mean_work_interactions_child   = 6,
                    mean_work_interactions_adult   = 12,
                    mean_work_interactions_elderly = 5,
                    daily_fraction_work            = 0.25,
                    work_network_rewire            = 0.3
                )
            ),
            dict(
                test_params = dict(
                    n_total = 1200,
                    end_time = 15,
                    mean_work_interactions_child   = 6,
                    mean_work_interactions_adult   = 12,
                    mean_work_interactions_elderly = 5,
                    daily_fraction_work            = 0.25,
                    work_network_rewire            = 0.3
                )
            ),
            dict(
                test_params = dict(
                    n_total = 200,
                    end_time = 15,
                    mean_work_interactions_child   = 6,
                    mean_work_interactions_adult   = 12,
                    mean_work_interactions_elderly = 5,
                    daily_fraction_work            = 0.25,
                    work_network_rewire            = 0.3
                )
            ),
            dict(
                test_params = dict(
                    n_total = 300,
                    end_time = 15,
                    mean_work_interactions_child   = 6,
                    mean_work_interactions_adult   = 12,
                    mean_work_interactions_elderly = 5,
                    daily_fraction_work            = 0.25,
                    work_network_rewire            = 0.3
                )
            ),
            dict(
                test_params = dict(
                    n_total = 200,
                    end_time = 15,
                    mean_work_interactions_child   = 6,
                    mean_work_interactions_adult   = 12,
                    mean_work_interactions_elderly = 5,
                    daily_fraction_work            = 0.25,
                    work_network_rewire            = 0.3
                )
            ),
            dict(
                test_params = dict(
                    n_total = 150,
                    end_time = 15,
                    mean_work_interactions_child   = 6,
                    mean_work_interactions_adult   = 12,
                    mean_work_interactions_elderly = 5,
                    daily_fraction_work            = 0.25,
                    work_network_rewire            = 0.3
                )
            ),
        ],
        "test_delete_network": [
            dict( 
                test_params = dict( 
                    n_total = 10000
                ),
                network_id = 0             
            ),
            dict( 
                test_params = dict( 
                    n_total = 10000
                ),
                network_id = 1             
            ),
            dict( 
                test_params = dict( 
                    n_total = 10000
                ),
                network_id = 2             
            ),
            dict( 
                test_params = dict( 
                    n_total = 10000
                ),
                network_id = "custom"            
            )
              
        ],
        "test_user_random_network": [
            dict( 
                test_params = dict( 
                    n_total = 10000
                ),
                p_network = 0.4,
                mean_conn = 5         
            ),
            dict( 
                test_params = dict( 
                    n_total = 10000
                ),
                p_network = 0.8,
                mean_conn = 2         
            ),
            dict( 
                test_params = dict( 
                    n_total = 10000
                ),
                p_network = 0.1,
                mean_conn = 10         
            )
        ],
        "test_get_network": [dict()],
        "test_static_network": [
            dict( 
                test_params = dict( 
                    n_total = 10000,
                    rebuild_networks = False,
                    infectious_rate = 10,
                    end_time = 20
                )
            )
        ],
        "test_network_transmission_multipler_update": [
            dict(
                test_params = dict( n_total = 10000 ),
                random_mult = 1.5,
                occupation_mult = 1.2,
                working_mult = 1.3,
                primary_mult = 0.7,
                house_mult  = 1.8   
            ),
            dict(
                test_params = dict( n_total = 10000 ),
                random_mult = 0.7,
                occupation_mult =0.3,
                working_mult = 3,
                primary_mult = 0.3,
                house_mult  = 0.8   
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
        
        params = utils.get_params_swig()
        params = utils.turn_off_interventions(params,1)
        params.set_param("n_total", n_total)
        
        model  = utils.get_model_swig( params )

        # Call the model
        for _ in range( params.get_param("end_time") ):
            model.one_time_step()
        
        # Write output files
        model.write_interactions_file()

        df_int = pd.read_csv( constant.TEST_INTERACTION_FILE )
        left  = df_int.loc[ :, ['ID_1', 'ID_2'] ]        
        right = df_int.loc[ :, ['ID_1', 'ID_2'] ]
        right.rename( columns =  {"ID_1":"ID_2","ID_2":"ID_1"},inplace=True)

        left.drop_duplicates(keep="first",inplace=True)
        right.drop_duplicates(keep="first",inplace=True)
        join = pd.merge(left,right,on=["ID_1","ID_2"], how="inner")
        
        N_base = len( left )
        N_join = len( join )
        
        np.testing.assert_equal( N_base, N_join )
    
    def test_household_network(self, n_total):
        """
        Test to check that all interactions within a household are made
        """
        # note when counting connections we count each end
        expectedConnections = pd.DataFrame(data={'size': [1,2,3,4,5,6], 'expected': [0,2,6,12,20,30]})
        
        params = utils.get_params_swig()
        params = utils.turn_off_interventions(params, 1)
        params.set_param("n_total", n_total)
        
        model  = utils.get_model_swig( params )

        # Call the model for one time step
        model.one_time_step()
        
        model.write_individual_file()
        model.write_interactions_file()

        # get the number of people in each house hold
        df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE )
        df_house = df_indiv.groupby(["house_no"]).size().reset_index(name="size")
     
        # get the number of interactions per person on the housegold n
        df_int  = pd.read_csv( constant.TEST_INTERACTION_FILE )
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
        sd_tolerance = 0.06
        
        # note when counting connections we count each end
        ageTypeMap = pd.DataFrame( data={
            "age_group": constant.AGES, 
            "age_type": constant.AGE_TYPES } )
        
        params = utils.get_params_swig()
        params = utils.turn_off_interventions(params, 1)
        params.set_param("mean_random_interactions_child",  mean_random_interactions_child )
        params.set_param("sd_random_interactions_child",    sd_random_interactions_child )
        params.set_param("mean_random_interactions_adult",  mean_random_interactions_adult )
        params.set_param("sd_random_interactions_adult",    sd_random_interactions_adult )
        params.set_param("mean_random_interactions_elderly",mean_random_interactions_elderly )
        params.set_param("sd_random_interactions_elderly",  sd_random_interactions_elderly )
        params.set_param("n_total",n_total)
        params.set_param("hospital_on", 0)
        
        model  = utils.get_model_swig( params )

        # Call the model for one time step
        model.one_time_step()
        
        model.write_individual_file()
        model.write_interactions_file()
        
        # get all the people, need to hand case if people having zero connections
        df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE )
        df_indiv = df_indiv.loc[:,["ID","age_group"]] 
        df_indiv = pd.merge( df_indiv, ageTypeMap, on = "age_group", how = "left" )

        # get all the random connections
        df_int = pd.read_csv( constant.TEST_INTERACTION_FILE )
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
        np.testing.assert_allclose( mean, mean_random_interactions_child, rtol = tolerance )
        if mean_random_interactions_child > 0:
            np.testing.assert_allclose( sd,     sd_random_interactions_child, rtol = sd_tolerance )
        
        mean = df_int[df_int["age_type"] == constant.ADULT].loc[:,"connections"].mean()
        sd   = df_int[df_int["age_type"] == constant.ADULT].loc[:,"connections"].std()        
        np.testing.assert_allclose( mean, mean_random_interactions_adult, rtol = tolerance )
        if mean_random_interactions_adult > 0:
            np.testing.assert_allclose( sd,   sd_random_interactions_adult, rtol = sd_tolerance )
        
        mean = df_int[df_int["age_type"] == constant.ELDERLY].loc[:,"connections"].mean()
        sd   = df_int[df_int["age_type"] == constant.ELDERLY].loc[:,"connections"].std()        
        np.testing.assert_allclose( mean, mean_random_interactions_elderly, rtol = tolerance )
        if mean_random_interactions_elderly > 0:
            np.testing.assert_allclose( sd,   sd_random_interactions_elderly, rtol = sd_tolerance )
  
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
        
        # note when counting connections we count each end
        ageTypeMap1 = pd.DataFrame( 
            data={ "age_group_1": constant.AGES, "age_type_1": constant.AGE_TYPES } )
        ageTypeMap2 = pd.DataFrame(
            data={ "age_group_2": constant.AGES, "age_type_2": constant.AGE_TYPES } )
            
        paramByNetworkType = [ 
            mean_work_interactions_child, 
            mean_work_interactions_adult, 
            mean_work_interactions_elderly ]
        
        params = utils.get_params_swig()
        params = utils.turn_off_interventions(params,1)
        params.set_param("n_total",n_total)
        params.set_param( "mean_work_interactions_child",   mean_work_interactions_child )
        params.set_param( "mean_work_interactions_adult",   mean_work_interactions_adult )
        params.set_param( "mean_work_interactions_elderly", mean_work_interactions_elderly )
        params.set_param( "daily_fraction_work",            daily_fraction_work )
        params.set_param( "n_total",n_total)
        params.set_param( "hospital_on", 0)
        
        model  = utils.get_model_swig( params )

        # Call the model for one time step
        model.one_time_step()

        model.write_individual_file()
        model.write_interactions_file()

        # get all the people, need to hand case if people having zero connections
        df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE )
        df_indiv = df_indiv.loc[:,[ "ID", "age_group", "occupation_network" ] ] 
        df_indiv = pd.merge( df_indiv, ageTypeMap1, 
            left_on = "age_group", right_on = "age_group_1", how = "left" )

        # get all the work connections
        df_int  = pd.read_csv(constant.TEST_INTERACTION_FILE )
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
        np.testing.assert_equal( n, 0, "only 80  adults on the 80 network" )
        
        # check the mean number of networks connections by network
        for network in [ constant.PRIMARY_NETWORK, constant.SECONDARY_NETWORK ]:
            mean = df_n_int[ df_n_int[ "occupation_network" ] == network ].loc[:,"connections"].mean()
            np.testing.assert_allclose( mean, mean_work_interactions_child, rtol = tolerance )
            
        mean = df_n_int[ df_n_int[ "occupation_network" ] == constant.WORKING_NETWORK ].loc[:,"connections"].mean()
        np.testing.assert_allclose( mean, mean_work_interactions_adult, rtol = tolerance )
        
        for network in [ constant.RETIRED_NETWORK, constant.ELDERLY_NETWORK ]:
            mean = df_n_int[ df_n_int[ "occupation_network" ] == network ].loc[:,"connections"].mean()
            np.testing.assert_allclose( mean, mean_work_interactions_elderly, rtol = tolerance )
      
        # check the correlation is below a threshold
        corr = df_int['house_no_1'].corr(df_int['house_no_2'])
        if ( len( df_int ) > 1 ) :
            np.testing.assert_allclose( corr, 0, atol = tolerance )

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
        tolerance = 0.02

        params = utils.get_params_swig()
        params = utils.turn_off_interventions(params,1)
        params.set_param("n_total",n_total)
        params.set_param( "child_network_adults",   child_network_adults )
        params.set_param( "elderly_network_adults",   elderly_network_adults )

        model  = utils.get_model_swig( params )

        # Call the model for one time step
        model.one_time_step()
        model.write_individual_file()

        # get all the people, need to hand case if people having zero connections
        df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE )
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
        model.one_time_step()
        model.write_interactions_file()
        df_inter = pd.read_csv(constant.TEST_INTERACTION_FILE)
        df_inter[ "time" ] = 0

        for time in range( test_params[ "end_time" ] ):
            model.one_time_step()
            model.write_interactions_file()
            df = pd.read_csv(constant.TEST_INTERACTION_FILE)
            df[ "time" ] = time + 1
            df_inter = pd.concat( [df_inter, df ])
  
        df_inter = df_inter[ df_inter[ "type" ] == constant.OCCUPATION ]
  
        # check to see there are sufficient daily connections and only one per set of contacts a day
        df_unique_daily = df_inter.groupby( ["time","ID_1","ID_2"]).size().reset_index(name="N")
        min_size = (test_params["end_time"]+1) * test_params[ "n_total"] *  min( 1,test_params["mean_work_interactions_child"],test_params["mean_work_interactions_adult"],test_params["mean_work_interactions_elderly"] )

        np.testing.assert_equal(sum(df_unique_daily["N"]==1)>min_size, True, "Less contacts than expected on the occuaptional networks" )
        np.testing.assert_equal(sum(df_unique_daily["N"]!=1), 0, "Repeat connections on same day on the occupational networks" )

        # check the mean unique connections over multiple days is mean/daily fraction
        df_unique = df_inter.groupby(["occupation_network_1","ID_1","ID_2"]).size().reset_index(name="N_unique")
        df_unique = df_unique.groupby(["occupation_network_1","ID_1"]).size().reset_index(name="N_conn")
        df_unique = df_unique.groupby(["occupation_network_1"]).mean()
    
        mean_by_type = [ test_params["mean_work_interactions_child"],test_params["mean_work_interactions_adult"],test_params["mean_work_interactions_elderly"]]
        
        for network in constant.NETWORKS:
            actual   = df_unique.loc[network]["N_conn"]
            expected = mean_by_type[constant.NETWORK_TYPE_MAP[network]]/test_params["daily_fraction_work"]
            np.testing.assert_allclose(actual,expected,rtol=tol,err_msg="Expected mean unique occupational contacts over multiple days not as expected")
           
    def test_user_defined_network(self, test_params, new_connections):
        """
            Adds in a user defined network with random connections
        """
        
        new_connections = int(new_connections)
         
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model   = utils.get_model_swig( params )
        n_total = params.get_param("n_total")
        
        # build a random network of new connections
        node_1 = [0]*new_connections
        node_2 = [0]*new_connections
        for i in range(new_connections):
            node_1[i] = rd.randrange(0,n_total-1)
            node_2[i] = rd.randrange(0,n_total-1)    
        network = pd.DataFrame({'ID_1':np.array(node_1, dtype='int32'), 'ID_2':np.array(node_2, dtype='int32')})
       
        # remove duplicates and edges connecting to self
        network = network[ (network["ID_1"] != network["ID_2"]) ]
        network = network.groupby(["ID_1","ID_2"]).size().reset_index(name = "count")
        n_edges = len( network )
        np.testing.assert_equal( n_edges > new_connections*0.9,True, err_msg = "In sufficient edges on the network to test")

        # add to the model and run
        model.add_user_network( network, name = "my test network",skip_hospitalised=False,skip_quarantine=False)
        for time in range( test_params[ "end_time" ] ):
            model.one_time_step()
            
        # get the interaction file and check the connections are there
        model.write_interactions_file()
        df_inter = pd.read_csv(constant.TEST_INTERACTION_FILE)
        df_inter[ "in_interaction"] = True
        
        # now check that all the edges in the network are interactions in the model
        df = pd.merge(network,df_inter,on = ["ID_1", "ID_2"], how = "left")
        n_miss = sum( df["in_interaction"] != True )
                
        np.testing.assert_equal( n_miss, 0, err_msg = "interactions from user network are missing")

    def test_custom_occupation_network( self, test_params ):
        """
          For user defined occupational networks,
          check to see that people only meet with the same person
          once per day on each occupational network;

          Check to see that when you look over multiple days that
          the mean number of unique contacts is mean_daily/daily_fraction
        """

        # Set up user-defined occupation network tables.
        n_total = test_params['n_total']
        IDs = np.arange(n_total, dtype='int32')
        network_no = np.arange(10, dtype='int32')  # Set up 10 occupation networks

        assignment = np.zeros(n_total, dtype='int32')
        for i in range(10):
            assignment[i*n_total//10 : (i+1)*n_total//10] = i

        age_type = np.zeros(10)
        age_type[0:2] = constant.CHILD
        age_type[2:8] = constant.ADULT
        age_type[8:10] = constant.ELDERLY

        mean_work_interaction = np.zeros(10)
        mean_work_interaction[0:2] = test_params['mean_work_interactions_child']
        mean_work_interaction[2:8] = test_params['mean_work_interactions_adult']
        mean_work_interaction[8:] = test_params['mean_work_interactions_elderly']

        lockdown_multiplier = np.ones(10) * 0.2

        network_name = ['primary', 'secondary', 'adult_1', 'adult_2', 'adult_3', 'adult_4',
                        'adult_5', 'adult_6', 'elderly_1', 'elderly_2']

        df_occupation_network  = pd.DataFrame({'ID':IDs,'network_no':assignment})
        df_occupation_network_property = pd.DataFrame({
            'network_no': network_no,
            'age_type': age_type,
            'mean_work_interaction': mean_work_interaction,
            'lockdown_multiplier': lockdown_multiplier,
            'network_name': network_name})

        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )
        # load custom occupation network table before constructing the model
        params.set_occupation_network_table(df_occupation_network, df_occupation_network_property)

        # make a simple demographic table. For small networks, household rejection sampling won't converge.
        hhIDs      = np.array( range(n_total), dtype='int32')
        house_no = np.array( hhIDs / 4, dtype='int32' )
        ages     = np.array( np.mod( hhIDs, 9) , dtype='int32' )
        df_demo  = pd.DataFrame({'ID': hhIDs,'age_group':ages,'house_no':house_no})

        # add to the parameters and get the model
        params.set_demographic_household_table( df_demo ),

        model  = utils.get_model_swig( params )
        model.one_time_step()
        model.write_interactions_file()
        df_inter = pd.read_csv(constant.TEST_INTERACTION_FILE)
        df_inter[ "time" ] = 0

        for time in range( test_params[ "end_time" ] ):
            model.one_time_step()
            model.write_interactions_file()
            df = pd.read_csv(constant.TEST_INTERACTION_FILE)
            df[ "time" ] = time + 1
            df_inter = pd.concat( [df_inter, df] )

        df_inter = df_inter[ list( df_inter[ "type" ] == constant.OCCUPATION ) ]

        # check to see there are sufficient daily connections and only one per set of contacts a day
        df_unique_daily = df_inter.groupby( ["time","ID_1","ID_2"]).size().reset_index(name="N")

        connection_upper_bound = (test_params["n_total"] / 10 -1 ) // 2 * 2.0

        min_size = (test_params["end_time"]+1) * test_params["n_total"] * (
                0.2 * min (connection_upper_bound * test_params['daily_fraction_work'], test_params["mean_work_interactions_child"] )
                + 0.6 * min (connection_upper_bound * test_params['daily_fraction_work'], test_params["mean_work_interactions_adult"])
                + 0.2 * min (connection_upper_bound * test_params['daily_fraction_work'], test_params["mean_work_interactions_elderly"]))

        np.testing.assert_allclose(sum(df_unique_daily["N"]==1), min_size, rtol=0.1, err_msg="Unexpected contacts on the occupational networks" )
        np.testing.assert_equal(sum(df_unique_daily["N"]!=1), 0, "Repeat connections on same day on the occupational networks" )

        # check the mean unique connections over multiple days is mean/daily fraction
        df_unique = df_inter.groupby(["occupation_network_1","ID_1","ID_2"]).size().reset_index(name="N_unique")
        df_unique = df_unique.groupby(["occupation_network_1","ID_1"]).size().reset_index(name="N_conn")
        df_unique = df_unique.groupby(["occupation_network_1"]).mean()

        mean_by_type = [ test_params["mean_work_interactions_child"],test_params["mean_work_interactions_adult"],test_params["mean_work_interactions_elderly"]]

        for network in range(10): # 10 custom occupation networks
            actual   = df_unique.loc[network]["N_conn"]
            expected = min( connection_upper_bound,
                            mean_by_type[constant.CUSTOM_NETWORK_TYPE_MAP[network]]/test_params["daily_fraction_work"])
            if expected == connection_upper_bound:
                atol = 1
                np.testing.assert_allclose(actual,expected,atol=atol,err_msg="Expected mean unique occupational contacts over multiple days not as expected")
            else:
                rtol = 0.02
                np.testing.assert_allclose(actual,expected,rtol=rtol,err_msg="Expected mean unique occupational contacts over multiple days not as expected")

    def test_delete_network( self, test_params, network_id ):
        """
        Check to see whether after a network is deleted there are no interactions on it
        """
        
        # set up test model
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model = utils.get_model_swig( params )    
        
        # add a custom network to check for interactions
        n_total = test_params[ "n_total" ]   
        df_net = pd.DataFrame({
            'ID' : range( n_total ),
            'N'  : np.repeat( 2, n_total )
        } )      
        custom_network = model.add_user_network_random( df_net, name = "custom network" )
        
        # if checking the custom network then use its id
        if network_id == "custom" :
            network_id = custom_network.network_id()
                   
        # get the interactions before deleting the network      
        model.one_time_step()
        model.write_interactions_file()
        df_inter = pd.read_csv(constant.TEST_INTERACTION_FILE)

        # check there are interactions on the interaction network
        n_inter = len( df_inter[ df_inter[ "network_id"] == network_id ] )
        np.testing.assert_equal( n_inter > test_params[ "n_total" ], True, err_msg = "In sufficient interactions on network to test" )
        df_old = df_inter.groupby( "network_id").size().reset_index(name="n_old")
        df_old = df_old[ df_old[ "network_id" ] != network_id ]
       
        # now delete the network 
        network = model.get_network_by_id( network_id )
        model.delete_network( network )
        
        # check there are no new interactions onthe network
        model.one_time_step()
        model.write_interactions_file()
        df_inter = pd.read_csv(constant.TEST_INTERACTION_FILE)
        n_inter = len( df_inter[ df_inter[ "network_id"] == network_id ] )
        np.testing.assert_equal( n_inter , 0, err_msg = "Interactions on the deleted network" )
        df_new = df_inter.groupby( "network_id").size().reset_index(name="n_new")
    
        # finally check we have not deleted other networks
        df_comp = pd.merge( df_old, df_new, on = "network_id", how = "outer" )
        
        # check we have not added/deleted any networks
        np.testing.assert_equal( df_comp[ "n_old"].isna().sum(), 0, err_msg = "Added new network iteractions" )
        np.testing.assert_equal( df_comp[ "n_new"].isna().sum(), 0, err_msg = "Removed old network iteractions" )
        np.testing.assert_allclose( df_comp[ "n_old"], df_comp[ "n_new"], rtol = 0.1, err_msg = "Number of interactions onnetwork changed by too much")
                   
    def test_user_random_network( self, test_params, p_network, mean_conn ):
         
        """
        Check a user specificed random network is correct
        
            p_network - probability someone is on the custom network
            mean_conn - mean number of connections for someone on the network
        """
        
        # set the np seed so the results are reproducible
        np.random.seed(0)
        
        # set up test model
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model = utils.get_model_swig( params )    
        
        # add a custom network to check for interactions
        n_total = test_params[ "n_total" ]   
        df_net = pd.DataFrame({
            'ID' : range( n_total ),
            'inc': np.random.binomial(np.ones(n_total, dtype=int), np.ones(n_total) * p_network),
            'N'  : np.random.geometric( p = 1 / mean_conn, size = n_total )
        } )   
        df_net = df_net[ list( ( df_net[ "inc"] == 1 ) & ( df_net[ "N"] > 0 ) ) ]   
        custom_network = model.add_user_network_random( df_net, name = "custom network" )
        network_id     = custom_network.network_id()
        
        # step forward one time step and get the connections
        model.one_time_step()
        model.write_interactions_file() 
        df_inter = pd.read_csv(constant.TEST_INTERACTION_FILE)
        n_inter = df_inter[ list( df_inter[ "network_id"] == network_id ) ].groupby( "ID_1").size().reset_index(name="N_inter")
        n_inter.rename( columns =  {"ID_1":"ID"},inplace=True)
        
        # check they are as expected
        df_comb = pd.merge( n_inter, df_net, on = "ID", how = "outer")
        np.testing.assert_equal( df_comb[ "N"].isna().sum() <= 1, True, err_msg = "Extra people on network" )
        np.testing.assert_equal( df_comb[ "N_inter"].isna().sum() <= 1, True, err_msg = "Missing people from network" )
        np.testing.assert_equal( np.sum( np.abs( df_comb[ "N"] -  df_comb[ "N_inter"] ) > 0 ) <= 1, True, err_msg = "People with incorrect number of interactions" )    
        
        # step forward one time step and get the connections 
        model.one_time_step()
        model.write_interactions_file() 
        df_inter2 = pd.read_csv(constant.TEST_INTERACTION_FILE)
        n_inter2 = df_inter2[ list( df_inter2[ "network_id"] == network_id ) ].groupby( "ID_1").size().reset_index(name="N_inter2")
        n_inter2.rename( columns =  {"ID_1":"ID"},inplace=True)
        
        # check they are as expected
        df_comb = pd.merge( df_comb, n_inter2, on = "ID", how = "outer")
        np.testing.assert_equal( df_comb[ "N"].isna().sum() <= 1, True, err_msg = "Extra people on network" )
        np.testing.assert_equal( df_comb[ "N_inter2"].isna().sum() <= 1, True, err_msg = "Missing people from network" )
        np.testing.assert_equal( np.sum( np.abs( df_comb[ "N"] -  df_comb[ "N_inter2"] ) > 0 ) <= 1, True, err_msg = "People with incorrect number of interactions" )
 
        # check the random connections has changed
        df_inter = df_inter[ list( df_inter[ "network_id"] == network_id ) ].loc[:,list({"ID_1", "ID_2"})]
        df_inter["step1"] = True
        df_inter2 = df_inter2[ list( df_inter2[ "network_id"] == network_id ) ].loc[:,list({"ID_1", "ID_2"})]
        df_inter2["step2"] = True
        inter = pd.merge( df_inter, df_inter2, on = ["ID_1", "ID_2"], how = "outer" )
        inter = inter.replace( 'NaN', False )
        inter[ "same"] = ( inter["step2"] == inter["step1"] )
        n_same = np.sum( inter[ "same" ]== True ) 
        n_tot  = len( inter )
        np.testing.assert_allclose( [ n_same / n_tot ], [ 0 ], atol = 0.05, err_msg = "Too many repeated random contacts" )
    
    def test_get_network(self):

        # Set up test model
        params = utils.get_params_swig()
        model = utils.get_model_swig( params )
        model.one_time_step()
        
        # Write network to file
        model.write_household_network()
        df_network_written = pd.read_csv(constant.TEST_HOUSEHOLD_NETWORK_FILE)

        # get the network_id for the household table from the info
        network_info = model.get_network_info()
        network_id = network_info[ network_info[ "type"] == constant.HOUSEHOLD ].iloc[0]["id"]

        # Return network as a dataframe
        network = model.get_network_by_id( network_id )
        df_network_returned = network.get_network()

        np.testing.assert_array_equal( 
            df_network_returned.to_numpy(), 
            df_network_written.to_numpy() 
            )
        
    def test_static_network(self, test_params ):
        
        """
        Tests static networks
        
        Check to see the interactions are the same at the start and end of a simulation
        Check that transmissions only occur amongst these intections
        """
        
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model = utils.get_model_swig( params )
        
        # get the intitial interactions
        model.one_time_step()
        model.write_interactions_file() 
        inter_initial = pd.read_csv(constant.TEST_INTERACTION_FILE)
        
        # run to the end
        model.run( verbose = False )
        model.write_interactions_file() 
        model.write_transmissions()  
        inter_final   = pd.read_csv( constant.TEST_INTERACTION_FILE )
        transmissions = pd.read_csv( constant.TEST_TRANSMISSION_FILE )
        
        # first check that the interactions don't change during the simulation
        np.testing.assert_( len( inter_initial ) > 100000, "insufficient interactions formed" )
        np.testing.assert_( len( inter_initial ) == len( inter_final ), "change in number of interactions during simulation" )
        
        inter_initial = inter_initial.groupby( ["ID_1", "ID_2" ] ).size().reset_index()
        inter_final = inter_initial.groupby( ["ID_1", "ID_2" ] ).size().reset_index()
       
        inter_shared = pd.merge( inter_initial, inter_final, on = ["ID_1", "ID_2" ], how = "inner" )
        np.testing.assert_( len( inter_initial ) == len( inter_shared ), "not all interactions at the beginning and end of simulation" )
        
        # check that transmissions are only passed via these interactions
        transmissions = transmissions[ transmissions[ "time_infected" ] > 0 ]
        transmissions.rename( columns = { "ID_recipient":"ID_1", "ID_source":"ID_2" }, inplace = True )
        transmissions_inter = pd.merge( transmissions, inter_initial, on = [ "ID_1", "ID_2" ], how = "inner")
        np.testing.assert_( len( transmissions ) > 100, "insufficient interactions to test" )
        np.testing.assert_( len( transmissions ) == len( transmissions_inter ), "transmissions not on an initial interaction" )
    
    def test_network_transmission_multipler_update(self, test_params, random_mult, occupation_mult, working_mult, primary_mult, house_mult ): 
    
        """
        Test network transmission multiplier
        
        Check to see the network transmission multipliers can be updated correctly
        """
        
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model = utils.get_model_swig( params )
        model.run( n_steps = 3, verbose = False )
        
        model.update_running_params( "relative_transmission_occupation", occupation_mult )
        model.update_running_params( "relative_transmission_random", random_mult )
        model.update_running_params( "relative_transmission_household", house_mult )
        
        # get the network objects
        network_info = model.get_network_info().sort_values(["id"])
        network_names = network_info[ "name"]
        house_idx   = [i for i, s in enumerate(network_names) if 'Household' in s][0]
        working_idx = [i for i, s in enumerate(network_names) if 'working' in s][0]
        primary_idx = [i for i, s in enumerate(network_names) if 'primary' in s][0]
        random_idx  = [i for i, s in enumerate(network_names) if 'Random' in s][0]
        
        house_network   = model.get_network_by_id( house_idx )
        working_network = model.get_network_by_id( working_idx )
        primary_network = model.get_network_by_id( primary_idx )
        random_network  = model.get_network_by_id( random_idx )
        
        # update the multipliers
        house_network.set_network_transmission_multiplier( house_mult )
        working_network.set_network_transmission_multiplier( working_mult )
        primary_network.set_network_transmission_multiplier( primary_mult )
        random_network.set_network_transmission_multiplier( random_mult )
        
        np.testing.assert_approx_equal(house_network.transmission_multiplier(), house_mult,
                                       err_msg = "household network transmission multiplier incorrectly set")  
        np.testing.assert_approx_equal(working_network.transmission_multiplier(), working_mult,
                                       err_msg = "working network transmission multiplier incorrectly set")  
        np.testing.assert_approx_equal(primary_network.transmission_multiplier(), primary_mult,
                                       err_msg = "primary network transmission multiplier incorrectly set")        
        np.testing.assert_approx_equal(random_network.transmission_multiplier(), random_mult,
                                       err_msg = "random network transmission multiplier incorrectly set")  
         
        np.testing.assert_approx_equal(house_network.transmission_multiplier_type(), house_mult,
                                       err_msg = "household network transmission multiplier incorrectly set")    
        np.testing.assert_approx_equal(working_network.transmission_multiplier_type(), occupation_mult,
                                       err_msg = "working network transmission multiplier incorrectly set")  
        np.testing.assert_approx_equal(primary_network.transmission_multiplier_type(), occupation_mult,
                                       err_msg = "primary network transmission multiplier incorrectly set")        
        np.testing.assert_approx_equal(random_network.transmission_multiplier_type(), random_mult,
                                       err_msg = "random network transmission multiplier incorrectly set")  
       
        np.testing.assert_approx_equal(house_network.transmission_multiplier_combined(), house_mult * house_mult,
                                       err_msg = "household network transmission multiplier incorrectly set")  
        np.testing.assert_approx_equal(working_network.transmission_multiplier_combined(), occupation_mult * working_mult,
                                       err_msg = "working network transmission multiplier incorrectly set")  
        np.testing.assert_approx_equal(primary_network.transmission_multiplier_combined(), occupation_mult * primary_mult,
                                       err_msg = "primary network transmission multiplier incorrectly set")        
        np.testing.assert_approx_equal(random_network.transmission_multiplier_combined(), random_mult * random_mult,
                                       err_msg = "random network transmission multiplier incorrectly set")  
       
        model.run( n_steps = 1, verbose = False )
         
        # turn on a lockdown and check the household multipler changes correctly
        model.update_running_params( "lockdown_on", True )
        lockdown_mult = params.get_param( "lockdown_house_interaction_multiplier" )
        np.testing.assert_approx_equal(house_network.transmission_multiplier_combined(), house_mult * house_mult * lockdown_mult,
                                       err_msg = "household network transmission multiplier incorrectly set")  
        model.run( n_steps = 1, verbose = False )
        
        model.update_running_params( "lockdown_on", False )
        np.testing.assert_approx_equal(house_network.transmission_multiplier_combined(), house_mult * house_mult,
                                       err_msg = "household network transmission multiplier incorrectly set")  
        
        
        
        
        
                   