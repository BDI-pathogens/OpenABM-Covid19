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
from os.path import join
import numpy as np, pandas as pd
from scipy import optimize
from math import sqrt, log, exp

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
                n_total                          = 10000
            ),
            dict( 
                mean_random_interactions_child   = 1,
                sd_random_interactions_child     = 2,
                mean_random_interactions_adult   = 1,
                sd_random_interactions_adult     = 2,
                mean_random_interactions_elderly = 1,
                sd_random_interactions_elderly   = 2,
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
        "test_work_network": [ 
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
        
        left  = df_int.loc[ :, ['ID', 'ID_2'] ]        
        right = df_int.loc[ :, ['ID', 'ID_2'] ]
        right.rename( columns =  {"ID":"ID_2","ID_2":"ID"},inplace=True)

        left.drop_duplicates(keep="first",inplace=True)
        right.drop_duplicates(keep="first",inplace=True)
        join = pd.merge(left,right,on=["ID","ID_2"], how="inner")
        
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
        np.testing.assert_array_equal( df_int.loc[:,"house_no"], df_int.loc[:,"house_no_2"] )

        df_int  = df_int.groupby(["house_no"]).size().reset_index(name="connections")
        
        # see whether that is the expected number
        df_house = pd.merge( df_house,expectedConnections,on =[ "size" ], how = "left")
        df_house = pd.merge( df_house,df_int, on =[ "house_no" ], how = "outer" )

        # check single person household without connections
        N_nocon        = df_house.loc[:,["connections"]].isnull().sum().sum()
        N_nocon_single = df_house[ df_house["size"] == 1].loc[ :,["connections"]].isnull().sum().sum()
        np.testing.assert_equal(  N_nocon, N_nocon_single )
        
        # check the rest are the same 
        df_house = df_house[ df_house["size"] > 1 ]
        print( df_house.head())
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
        ageTypeMap = pd.DataFrame( data={ "age_group": constant.AGES, "age_type": constant.AGE_TYPES } );
                
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
        corr = df_int['house_no'].corr(df_int['house_no_2'])
        if ( len( df_int ) > 1 ) :
            np.testing.assert_allclose( corr, 0, atol = tolerance )
        
        df_int = df_int.loc[:,["ID"]] 
        df_int = df_int.groupby(["ID"]).size().reset_index(name="connections")
        df_int = pd.merge( df_indiv, df_int, on = "ID", how = "left" )
        df_int.fillna(0,inplace=True)
        
        # check mean and 
        mean = df_int[df_int["age_type"] == constant.CHILD].loc[:,"connections"].mean()
        sd   = df_int[df_int["age_type"] == constant.CHILD].loc[:,"connections"].std()        
        np.testing.assert_allclose( mean,   mean_random_interactions_child, rtol = tolerance )
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
  
    def test_work_network( 
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
        ageTypeMap         = pd.DataFrame( data={ "age_group": constant.AGES, "age_type": constant.AGE_TYPES } );
        ageTypeMap2        = pd.DataFrame( data={ "age_group_2": constant.AGES, "age_type_2": constant.AGE_TYPES } );
        paramByNetworkType = [ mean_work_interactions_child, mean_work_interactions_adult, mean_work_interactions_elderly ]      
        
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params = utils.turn_off_interventions(params,1)
        params.set_param("n_total",n_total)
        params.set_param( "mean_work_interactions_child",   mean_work_interactions_child )
        params.set_param( "mean_work_interactions_adult",   mean_work_interactions_adult )
        params.set_param( "mean_work_interactions_elderly", mean_work_interactions_elderly )
        params.set_param( "daily_fraction_work",            daily_fraction_work )
        params.set_param( "n_total",n_total)
        params.write_params(constant.TEST_DATA_FILE)        

        file_output   = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
       
        # get all the people, need to hand case if people having zero connections
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE, comment = "#", sep = ",", skipinitialspace = True )
        df_indiv = df_indiv.loc[:,[ "ID", "age_group", "work_network" ] ] 
        df_indiv = pd.merge( df_indiv, ageTypeMap, on = "age_group", how = "left" )

        # get all the work connections
        df_int  = pd.read_csv(constant.TEST_INTERACTION_FILE, comment = "#", sep = ",", skipinitialspace = True )
        df_int  = df_int[ df_int["type"] == constant.WORK ]
        df_int = pd.merge( df_int, ageTypeMap,  on = "age_group", how = "left" )
        df_int = pd.merge( df_int, ageTypeMap2, on = "age_group_2", how = "left" )

        # get the number of connections for each person
        df_n_int = df_int.groupby( [ "ID" ] ).size().reset_index( name = "connections" )
        df_n_int = pd.merge( df_indiv, df_n_int, on = "ID", how = "left" )
        df_n_int.fillna( 0, inplace=True )

        # check there are connections for each age group
        for age in constant.AGES:
            if ( paramByNetworkType[ constant.NETWORK_TYPES[ constant.AGE_TYPES[ age ] ] ]  > 0 ) :
                n = sum( df_int[ "age_group" ] == age )
                np.testing.assert_equal( n > 0, True, "there are no work connections for age_group " + str( age ) )
           
        # check the correct people are on each network 
        n = sum( 
            ( df_int[ "age_group" ] == constant.AGE_0_9 ) & 
            ( df_int[ "age_group_2" ] != constant.AGE_0_9 ) & 
            ( df_int[ "age_type_2" ] != constant.ADULT ) 
            )
        np.testing.assert_equal( n, 0, "only 0_9 and adults on the 0_9 network" )
        
        n = sum( 
            ( df_int[ "age_group" ] == constant.AGE_10_19 ) & 
            ( df_int[ "age_group_2" ] != constant.AGE_10_19 ) & 
            ( df_int[ "age_type_2" ] != constant.ADULT ) 
            )
        np.testing.assert_equal( n, 0, "only 10_19 and adults on the 10_19 network" )
        
        n = sum( 
            ( df_int[ "age_group" ] == constant.AGE_70_79 ) & 
            ( df_int[ "age_group_2" ] != constant.AGE_70_79 ) & 
            ( df_int[ "age_type_2" ] != constant.ADULT ) 
            )
        np.testing.assert_equal( n, 0, "only 70_79 and adults on the 70_79 network" )
        
        n = sum( 
            ( df_int[ "age_group" ] == constant.AGE_80 ) & 
            ( df_int[ "age_group_2" ] != constant.AGE_80 ) & 
            ( df_int[ "age_type_2" ] != constant.ADULT ) 
            )
        np.testing.assert_equal( n, 0, "only 80  adults on the 80 network" )
        
        # check the mean number of networks connections by network
        for network in [ constant.NETWORK_0_9, constant.NETWORK_10_19 ]:
            mean = df_n_int[ df_n_int[ "work_network" ] == network ].loc[:,"connections"].mean()
            np.testing.assert_allclose( mean, mean_work_interactions_child, rtol = tolerance )
            
        mean = df_n_int[ df_n_int[ "work_network" ] == constant.NETWORK_20_69 ].loc[:,"connections"].mean()
        np.testing.assert_allclose( mean, mean_work_interactions_adult, rtol = tolerance )
        
        for network in [ constant.NETWORK_70_79, constant.NETWORK_80 ]:
            mean = df_n_int[ df_n_int[ "work_network" ] == network ].loc[:,"connections"].mean()
            np.testing.assert_allclose( mean, mean_work_interactions_elderly, rtol = tolerance )
      
        # check the correlation is below a threshold
        corr = df_int['house_no'].corr(df_int['house_no_2'])
        if ( len( df_int ) > 1 ) :
            np.testing.assert_allclose( corr, 0, atol = tolerance )
