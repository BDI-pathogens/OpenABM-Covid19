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
IBM_DIR       = "src"
IBM_DIR_TEST  = "src_test"
DATA_DIR_TEST = "data_test"

TEST_DATA_TEMPLATE = "./tests/data/baseline_parameters.csv"
TEST_DATA_FILE     = join(DATA_DIR_TEST, "test_parameters.csv")

TEST_OUTPUT_FILE      = join(DATA_DIR_TEST, "test_output.csv")
TEST_INDIVIDUAL_FILE  = join(DATA_DIR_TEST, "individual_file_Run1.csv")
TEST_INTERACTION_FILE = join(DATA_DIR_TEST, "interactions_Run1.csv")

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
                n_total = 10000,
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
        df_individual = pd.read_csv(TEST_INTERACTION_FILE, comment = "#", sep = ",")
        
        np.testing.assert_equal(df_individual.shape[0] > 1, True)


    def test_network_connections_2way(self,n_total):
        """
        Test to check that for all connections in every persons interaction diary
        there is a corresponding connection in the other person's diary 
        """        
        params = ParameterSet(TEST_DATA_FILE, line_number = 1)
        params.set_param("n_total",n_total)
        utils.turn_off_interventions(params,1)  
        params.write_params(TEST_DATA_FILE)        
      
        file_output   = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command], stdout = file_output, shell = True)
        df_int        = pd.read_csv(TEST_INTERACTION_FILE, comment = "#", sep = ",", skipinitialspace = True )
        
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
            
        params = ParameterSet(TEST_DATA_FILE, line_number = 1)
        params.set_param("n_total",n_total)
        utils.turn_off_interventions(params,1)
        params.write_params(TEST_DATA_FILE)        

        file_output   = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command], stdout = file_output, shell = True)
       
        # get the number of people in each house hold
        df_indiv = pd.read_csv(TEST_INDIVIDUAL_FILE, comment = "#", sep = ",", skipinitialspace = True )
        df_house = df_indiv.groupby(["house_no"]).size().reset_index(name="size")
     
        # get the number of interactions per person on the housegold n
        df_int  = pd.read_csv(TEST_INTERACTION_FILE, comment = "#", sep = ",", skipinitialspace = True )
        df_int  = df_int[ df_int["type"] == HOUSEHOLD]
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
        ageTypeMap = pd.DataFrame( data={ "age_group": AGES, "age_type": AGE_TYPES } );
                
        params = ParameterSet(TEST_DATA_FILE, line_number = 1)
        params.set_param("mean_random_interactions_child",  mean_random_interactions_child )
        params.set_param("sd_random_interactions_child",    sd_random_interactions_child )
        params.set_param("mean_random_interactions_adult",  mean_random_interactions_adult )
        params.set_param("sd_random_interactions_adult",    sd_random_interactions_adult )
        params.set_param("mean_random_interactions_elderly",mean_random_interactions_elderly )
        params.set_param("sd_random_interactions_elderly",  sd_random_interactions_elderly )
        params.set_param("n_total",n_total)
        utils.turn_off_interventions(params,1)
        params.write_params(TEST_DATA_FILE)        

        file_output   = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command], stdout = file_output, shell = True)
       
        # get all the people, need to hand case if people having zero connections
        df_indiv = pd.read_csv(TEST_INDIVIDUAL_FILE, comment = "#", sep = ",", skipinitialspace = True )
        df_indiv = df_indiv.loc[:,["ID","age_group"]] 
        df_indiv = pd.merge( df_indiv, ageTypeMap, on = "age_group", how = "left" )

        # get all the random connections
        df_int = pd.read_csv(TEST_INTERACTION_FILE, comment = "#", sep = ",", skipinitialspace = True )
        df_int = df_int[ df_int["type"] == RANDOM ]
        
        df_int = df_int.loc[:,["ID"]] 
        df_int = df_int.groupby(["ID"]).size().reset_index(name="connections")
        df_int = pd.merge( df_indiv, df_int, on = "ID", how = "left" )
        df_int.fillna(0,inplace=True)
        
        # check mean and 
        mean = df_int[df_int["age_type"] == CHILD].loc[:,"connections"].mean()
        sd   = df_int[df_int["age_type"] == CHILD].loc[:,"connections"].std()        
        np.testing.assert_allclose( mean,   mean_random_interactions_child, rtol = tolerance )
        if mean_random_interactions_child > 0:
            np.testing.assert_allclose( sd,     sd_random_interactions_child, rtol = tolerance )
        
        mean = df_int[df_int["age_type"] == ADULT].loc[:,"connections"].mean()
        sd   = df_int[df_int["age_type"] == ADULT].loc[:,"connections"].std()        
        np.testing.assert_allclose( mean, mean_random_interactions_adult, rtol = tolerance )
        if mean_random_interactions_adult > 0:
            np.testing.assert_allclose( sd,   sd_random_interactions_adult, rtol = tolerance )
        
        mean = df_int[df_int["age_type"] == ELDERLY].loc[:,"connections"].mean()
        sd   = df_int[df_int["age_type"] == ELDERLY].loc[:,"connections"].std()        
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
        tolerance = 0.03
        
        # note when counting connections we count each end
        ageTypeMap         = pd.DataFrame( data={ "age_group": AGES, "age_type": AGE_TYPES } );
        ageTypeMap2        = pd.DataFrame( data={ "age_group_2": AGES, "age_type_2": AGE_TYPES } );
        paramByNetworkType = [ mean_work_interactions_child, mean_work_interactions_adult, mean_work_interactions_elderly ]      
                
        params = ParameterSet(TEST_DATA_FILE, line_number = 1)
        params.set_param("n_total",n_total)

        params.set_param( "mean_work_interactions_child",   mean_work_interactions_child )
        params.set_param( "mean_work_interactions_adult",   mean_work_interactions_adult )
        params.set_param( "mean_work_interactions_elderly", mean_work_interactions_elderly )
        params.set_param( "daily_fraction_work",            daily_fraction_work )
        params.set_param( "n_total",n_total)
        utils.turn_off_interventions(params,1)
        params.write_params(TEST_DATA_FILE)        

        file_output   = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command], stdout = file_output, shell = True)
       
        # get all the people, need to hand case if people having zero connections
        df_indiv = pd.read_csv(TEST_INDIVIDUAL_FILE, comment = "#", sep = ",", skipinitialspace = True )
        df_indiv = df_indiv.loc[:,[ "ID", "age_group", "work_network" ] ] 
        df_indiv = pd.merge( df_indiv, ageTypeMap, on = "age_group", how = "left" )

        # get all the random connections
        df_int  = pd.read_csv(TEST_INTERACTION_FILE, comment = "#", sep = ",", skipinitialspace = True )
        df_int  = df_int[ df_int["type"] == WORK ]
        df_int = pd.merge( df_int, ageTypeMap,  on = "age_group", how = "left" )
        df_int = pd.merge( df_int, ageTypeMap2, on = "age_group_2", how = "left" )

        # get the number of connections for each person
        df_n_int = df_int.groupby( [ "ID" ] ).size().reset_index( name = "connections" )
        df_n_int = pd.merge( df_indiv, df_n_int, on = "ID", how = "left" )
        df_n_int.fillna( 0, inplace=True )

        # check there are connections for each age group
        for age in AGES:
            if ( paramByNetworkType[ NETWORK_TYPES[ AGE_TYPES[ age ] ] ]  > 0 ) :
                n = sum( df_int[ "age_group" ] == age )
                np.testing.assert_equal( n > 0, True, "there are no work connections for age_group " + str( age ) )
           
        # check the correct people are on each network 
        n = sum( ( df_int[ "age_group" ] == AGE_0_9 ) & ( df_int[ "age_group_2" ] != AGE_0_9 ) & ( df_int[ "age_type_2" ] != ADULT ) )
        np.testing.assert_equal( n, 0, "only 0_9 and adults on the 0_9 network" )
        n = sum( ( df_int[ "age_group" ] == AGE_10_19 ) & ( df_int[ "age_group_2" ] != AGE_10_19 ) & ( df_int[ "age_type_2" ] != ADULT ) )
        np.testing.assert_equal( n, 0, "only 10_19 and adults on the 10_19 network" )
        n = sum( ( df_int[ "age_group" ] == AGE_70_79 ) & ( df_int[ "age_group_2" ] != AGE_70_79 ) & ( df_int[ "age_type_2" ] != ADULT ) )
        np.testing.assert_equal( n, 0, "only 70_79 and adults on the 70_79 network" )
        n = sum( ( df_int[ "age_group" ] == AGE_80 ) & ( df_int[ "age_group_2" ] != AGE_80 ) & ( df_int[ "age_type_2" ] != ADULT ) )
        np.testing.assert_equal( n, 0, "only 80  adults on the 80 network" )
        
        # check the mean number of networks connections by network
        for network in [ NETWORK_0_9, NETWORK_10_19 ]:
            mean = df_n_int[ df_n_int[ "work_network" ] == network ].loc[:,"connections"].mean()
            np.testing.assert_allclose( mean, mean_work_interactions_child, rtol = tolerance )
            
        mean = df_n_int[ df_n_int[ "work_network" ] == NETWORK_20_69 ].loc[:,"connections"].mean()
        np.testing.assert_allclose( mean, mean_work_interactions_adult, rtol = tolerance )
        
        for network in [ NETWORK_70_79, NETWORK_80 ]:
            mean = df_n_int[ df_n_int[ "work_network" ] == network ].loc[:,"connections"].mean()
            np.testing.assert_allclose( mean, mean_work_interactions_elderly, rtol = tolerance )
      
            
  