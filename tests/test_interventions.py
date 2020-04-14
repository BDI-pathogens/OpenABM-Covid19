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

# from test.test_bufio import lengths
# from CoreGraphics._CoreGraphics import CGRect_getMidX


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )


class TestClass(object):
    params = {
        "test_zero_quarantine": [dict()],
        "test_hospitalised_zero": [dict()],
        "test_quarantine_interactions": [
            dict(
                test_params=dict(
                    n_total=50000,
                    quarantined_daily_interactions=0,
                    end_time=25,
                    infectious_rate=4,
                    self_quarantine_fraction=1.0,
                    daily_non_cov_symptoms_rate=0.0,
                )
            ),
            dict(
                test_params=dict(
                    n_total=50000,
                    quarantined_daily_interactions=1,
                    end_time=25,
                    infectious_rate=4,
                    self_quarantine_fraction=0.75,
                    daily_non_cov_symptoms_rate=0.0005,
                )
            ),
            dict(
                test_params=dict(
                    n_total=50000,
                    quarantined_daily_interactions=2,
                    end_time=25,
                    infectious_rate=4,
                    self_quarantine_fraction=0.50,
                    daily_non_cov_symptoms_rate=0.001,
                )
            ),
            dict(
                test_params=dict(
                    n_total=50000,
                    quarantined_daily_interactions=3,
                    end_time=25,
                    infectious_rate=4,
                    self_quarantine_fraction=0.25,
                    daily_non_cov_symptoms_rate=0.005,
                )
            ),
        ],
        "test_quarantine_on_symptoms": [
            dict(
                test_params=dict(
                    n_total=50000,
                    end_time=25,
                    infectious_rate=4,
                    self_quarantine_fraction=0.8,
                    daily_non_cov_symptoms_rate=0.0,
                    asymptomatic_infectious_factor=0.4,
                )
            ),
            dict(
                test_params=dict(
                    n_total=50000,
                    end_time=1,
                    infectious_rate=4,
                    self_quarantine_fraction=0.5,
                    daily_non_cov_symptoms_rate=0.05,
                    asymptomatic_infectious_factor=1.0,
                )
            ),
        ],
        "test_quarantine_household_on_symptoms": [ 
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 500,
                    end_time = 20,
                    infectious_rate = 4,
                    self_quarantine_fraction = 1.0,
                    quarantine_household_on_symptoms = 1
                )
            ) 
        ],
        "test_trace_on_symptoms": [ 
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 500,
                    end_time = 20,
                    infectious_rate = 4,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = 1,
                    quarantine_on_traced = 1,
                    app_turn_on_time = 1,
                    quarantine_household_on_symptoms = 1
                ),
                app_users_fraction = 0.85

            ) 
        ],
        "test_lockdown_transmission_rates": [ 
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 10000,
                    end_time = 3,
                    infectious_rate = 4,
                    lockdown_work_network_multiplier = 0.8,
                    lockdown_random_network_multiplier = 0.8,
                    lockdown_house_interaction_multiplier = 1.2
                )
            ) 
        ],
         "test_app_users_fraction": [ 
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 500,
                    end_time = 20,
                    infectious_rate = 4,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = 1,
                    quarantine_on_traced = 1,
                    app_turn_on_time = 1,
                    app_users_fraction_0_9 = 0,
                    app_users_fraction_10_19 = 0.8,
                    app_users_fraction_20_29 = 0.8,
                    app_users_fraction_30_39 = 0.8,
                    app_users_fraction_40_49 = 0.8,
                    app_users_fraction_50_59 = 0.8,
                    app_users_fraction_60_69 = 0.8,
                    app_users_fraction_70_79 = 0.4,
                    app_users_fraction_80 = 0.2,
                    traceable_interaction_fraction = 1
                ),

            ),
        ]
    }
    """
    Test class for checking 
    """
    def test_zero_quarantine(self):
        """
        Test there are no individuals quarantined if all quarantine parameters are "turned off"
        """
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params = utils.turn_off_quarantine(params)
        params.write_params(constant.TEST_DATA_FILE)
        
        # Call the model
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
        np.testing.assert_equal(df_output["n_quarantine"].to_numpy().sum(), 0)
    
    def test_hospitalised_zero(self):
        """
        Test setting hospitalised fractions to zero (should be no hospitalised)
        """
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params = utils.set_hospitalisation_fraction_all(params, 0.0)
        params.write_params(constant.TEST_DATA_FILE)
        
        # Call the model, pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        np.testing.assert_equal(df_output["n_hospital"].sum(), 0)
    
    def test_quarantine_interactions(self, test_params):
        """
        Tests the number of interactions people have on the interaction network is as 
        described when they have been quarantined
        """
        tolerance = 0.01
        end_time = test_params["end_time"]

        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params = utils.turn_off_interventions(params, end_time)
        params.set_param(test_params)
        params.write_params(constant.TEST_DATA_FILE)

        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout=file_output, shell=True)
        df_indiv = pd.read_csv(
            constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True
        )
        df_int = pd.read_csv(
            constant.TEST_INTERACTION_FILE, comment="#", sep=",", skipinitialspace=True
        )

        # get the people who are in quarantine and were on the previous step
        df_quar = df_indiv[
            (df_indiv["quarantined"] == 1) & (df_indiv["time_quarantined"] < end_time)
        ]
        df_quar = df_quar.loc[:, "ID"]

        # get the number of interactions by type
        df_int = df_int.groupby(["ID", "type"]).size().reset_index(name="connections")

        # check to see there are no work connections
        df_test = pd.merge(
            df_quar, df_int[df_int["type"] == constant.WORK], on="ID", how="inner"
        )
        np.testing.assert_equal(
            len(df_test), 0, "quarantined individual with work contacts"
        )

        # check to see there are are household connections
        df_test = pd.merge(
            df_quar, df_int[df_int["type"] == constant.HOUSEHOLD], on="ID", how="inner"
        )
        np.testing.assert_equal(
            len(df_test) > 0,
            True,
            "quarantined individuals have no household connections",
        )

        # check to whether the number of random connections are as specified
        df_test = pd.merge(
            df_quar, df_int[df_int["type"] == constant.RANDOM], on="ID", how="left"
        )
        df_test.fillna(0, inplace=True)
        np.testing.assert_allclose(
            df_test.loc[:, "connections"].mean(),
            float(params.get_param("quarantined_daily_interactions")),
            rtol=tolerance,
        )

    def test_quarantine_on_symptoms(self, test_params):
        """
        Tests the correct proportion of people are self-isolating on sypmtoms
        """

        tolerance = 0.01
        tol_sd = 4
        end_time = test_params["end_time"]

        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params = utils.turn_off_interventions(params, end_time)
        params.set_param(test_params)
        params.write_params(constant.TEST_DATA_FILE)

        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout=file_output, shell=True)
        df_indiv = pd.read_csv(
            constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True
        )
        df_int = pd.read_csv(
            constant.TEST_INTERACTION_FILE, comment="#", sep=",", skipinitialspace=True
        )

        # get the people who are in quarantine and were on the last step
        df_quar = df_indiv[
            (df_indiv["quarantined"] == 1) & (df_indiv["time_quarantined"] == end_time)
        ]
        df_quar = df_quar.loc[:, "ID"]
        n_quar = len(df_quar)

        # get the people who developed symptoms on the last step
        df_symp = df_indiv[(df_indiv["time_symptomatic"] == end_time)]
        n_symp = len(df_symp)

        # if no seasonal flu or contact tracing then this is the only path
        if test_params["daily_non_cov_symptoms_rate"] == 0:
            df = pd.merge(df_quar, df_symp, on="ID", how="inner")
            np.testing.assert_equal(
                n_quar,
                len(df),
                "people quarantined without symptoms when daily_non_cov_symptoms_rate turned off",
            )

            n_exp_quar = n_symp * test_params["self_quarantine_fraction"]
            np.testing.assert_allclose(
                n_exp_quar,
                n_quar,
                atol=tol_sd * sqrt(n_exp_quar),
                err_msg="the number of quarantined not explained by symptoms",
            )

        # if no symptomatic then check number of newly quarantined is from flu
        elif end_time == 1 and test_params["asymptomatic_infectious_factor"] == 1:
            n_flu = test_params["n_total"] * test_params["daily_non_cov_symptoms_rate"]
            n_exp_quar = n_flu * test_params["self_quarantine_fraction"]

            np.testing.assert_allclose(
                n_exp_quar,
                n_quar,
                atol=tol_sd * sqrt(n_exp_quar),
                err_msg="the number of quarantined not explained by daily_non_cov_symptoms_rate",
            )

        else:
            np.testing.assert_equal(
                True, False, "no test run due test_params not being testable"
            )
    def test_quarantine_household_on_symptoms(self, test_params ):
        """
        Tests households are quarantine when somebody has symptoms
        """
        end_time = test_params[ "end_time" ]

        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params = utils.turn_off_interventions(params, end_time)
        params.set_param(test_params)
        params.write_params(constant.TEST_DATA_FILE)

        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout=file_output, shell=True)
        df_int   = pd.read_csv( constant.TEST_INTERACTION_FILE, comment="#", sep=",", skipinitialspace=True )
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
        
        # prepare the interaction data to get all household interations
        df_int.rename( columns = { "ID":"index_ID", "ID_2":"traced_ID"}, inplace = True )
        df_int[ "household" ] = ( df_int[ "house_no" ] == df_int[ "house_no_2" ] )
        df_int = df_int.loc[ :, [ "index_ID", "traced_ID", "household"]]
                
        # don't consider ones with multiple index events
        filter_single = df_trace.groupby( ["index_ID", "days_since_index"] ).size();
        filter_single = filter_single.groupby( ["index_ID"]).size().reset_index(name="N");
        filter_single = filter_single[ filter_single[ "N"] == 1 ]
        
        # look at the trace token data to get all traces
        index_traced = df_trace[ ( df_trace[ "time" ] == end_time ) & ( df_trace[ "days_since_contact" ] == 0 ) ] 
        index_traced = index_traced.groupby( [ "index_ID", "traced_ID" ] ).size().reset_index(name="cons")    
        index_traced[ "traced" ] = True
        index_traced = pd.merge( index_traced, filter_single, on = "index_ID", how = "inner")
       
        # get all the interactions for the index cases
        index_cases  = pd.DataFrame( data = { 'index_ID': index_traced.index_ID.unique() } )
        index_inter = pd.merge( index_cases, df_int, on = "index_ID", how = "left" )             
        index_inter = index_inter.groupby( [ "index_ID", "traced_ID", "household" ]).size().reset_index(name="N")    
        index_inter[ "inter" ] = True

        # test nobody traced without an interaction
        t = pd.merge( index_traced, index_inter, on = [ "index_ID", "traced_ID" ], how = "outer" )
        n_no_inter = len( t[ t[ "inter"] != True ] )
        np.testing.assert_equal( n_no_inter, 0, "tracing someone without an interaction" )

        # check everybody with a household interaction is traced
        n_no_trace = len( t[ ( t[ "traced"] != True ) &  (t["household"] == True  )] )
        np.testing.assert_equal( n_no_trace, 0, "failed to trace someone in the household" )

    def test_lockdown_transmission_rates(self, test_params):
        """
        Tests the change in transmission rates on lockdown are correct
        NOTE - this can only be done soon after a random seed and for small
        changes due to saturation effects
        """
        
        sd_diff  = 3;
        end_time = test_params[ "end_time" ]

        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params = utils.turn_off_interventions(params, end_time)
        params.set_param(test_params)
        params.write_params(constant.TEST_DATA_FILE)
        
        # run without lockdown
        file_output   = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout=file_output, shell=True)
        df_without    = pd.read_csv( constant.TEST_TRANSMISSION_FILE, comment="#", sep=",", skipinitialspace=True )
        df_without    = df_without[ df_without[ "time_infected"] == end_time ].groupby( [ "infector_network"] ).size().reset_index(name="N")

        # lockdown on t-1
        params = utils.turn_off_interventions(params, end_time)
        params.set_param(test_params)
        params.write_params(constant.TEST_DATA_FILE)
        params.set_param( "lockdown_time_on", end_time - 1 );
        params.write_params(constant.TEST_DATA_FILE)
        
        file_output   = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout=file_output, shell=True)
        df_with       = pd.read_csv( constant.TEST_TRANSMISSION_FILE, comment="#", sep=",", skipinitialspace=True )
        df_with       = df_with[ df_with[ "time_infected"] == end_time ].groupby( [ "infector_network"] ).size().reset_index(name="N")
        
        # now check they are line
        expect_household = df_without.loc[ constant.HOUSEHOLD, ["N"] ] * test_params[ "lockdown_house_interaction_multiplier" ]       
        np.testing.assert_allclose( df_with.loc[ constant.HOUSEHOLD, ["N"] ], expect_household, atol = sqrt( expect_household ) * sd_diff, 
                                    err_msg = "lockdown not changing household transmission as expected" )
        
        expect_work = df_without.loc[ constant.WORK, ["N"] ] * test_params[ "lockdown_work_network_multiplier" ]       
        np.testing.assert_allclose( df_with.loc[ constant.WORK, ["N"] ], expect_work, atol = sqrt( expect_work) * sd_diff, 
                                    err_msg = "lockdown not changing work transmission as expected" )
      
      
        expect_random = df_without.loc[ constant.RANDOM, ["N"] ] * test_params[ "lockdown_random_network_multiplier" ]       
        np.testing.assert_allclose( df_with.loc[ constant.RANDOM, ["N"] ], expect_random, atol = sqrt( expect_random ) * sd_diff, 
                                    err_msg = "lockdown not changing random transmission as expected" )
      

    def test_trace_on_symptoms(self, test_params, app_users_fraction ):
        """
        Tests that people who are traced on symptoms are
        real contacts
        """
        end_time = test_params[ "end_time" ]

        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params = utils.turn_off_interventions(params, end_time)
        params = utils.set_app_users_fraction_all( params, app_users_fraction )
        params.set_param(test_params)
        params.write_params(constant.TEST_DATA_FILE)

        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout=file_output, shell=True)
        df_int   = pd.read_csv( constant.TEST_INTERACTION_FILE, comment="#", sep=",", skipinitialspace=True )
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )

        # prepare the interaction data to get all household interations
        df_int.rename( columns = { "ID":"index_ID", "ID_2":"traced_ID"}, inplace = True )
        df_int[ "household" ] = ( df_int[ "house_no" ] == df_int[ "house_no_2" ] )
        df_int = df_int.loc[ :, [ "index_ID", "traced_ID", "household"]]

        # don't consider ones with multiple index events
        filter_single = df_trace.groupby( ["index_ID", "days_since_index"] ).size();
        filter_single = filter_single.groupby( ["index_ID"]).size().reset_index(name="N");
        filter_single = filter_single[ filter_single[ "N"] == 1 ]

        # look at the trace token data to get all traces
        index_traced = df_trace[ ( df_trace[ "time" ] == end_time ) & ( df_trace[ "days_since_contact" ] == 0 ) ] 
        index_traced = index_traced.groupby( [ "index_ID", "traced_ID" ] ).size().reset_index(name="cons")    
        index_traced[ "traced" ] = True
        index_traced = pd.merge( index_traced, filter_single, on = "index_ID", how = "inner")

        # get all the interactions for the index cases
        index_cases  = pd.DataFrame( data = { 'index_ID': index_traced.index_ID.unique() } )
        index_inter = pd.merge( index_cases, df_int, on = "index_ID", how = "left" )             
        index_inter = index_inter.groupby( [ "index_ID", "traced_ID", "household" ]).size().reset_index(name="N")    
        index_inter[ "inter" ] = True

        # test nobody traced without an interaction
        t = pd.merge( index_traced, index_inter, on = [ "index_ID", "traced_ID" ], how = "outer" )
        n_no_inter = len( t[ t[ "inter"] != True ] )
        np.testing.assert_equal( n_no_inter, 0, "tracing someone without an interaction" )    

    
    def test_app_users_fraction(self, test_params ):
        """
        Tests that the correct number of people are assigned
        use the app and that only app users start tracing 
        and can be traced if household options are not turned on
        """
        end_time = test_params[ "end_time" ]

        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param(test_params)
        params.write_params(constant.TEST_DATA_FILE)

        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout=file_output, shell=True)
        df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )
        
        app_users  = df_indiv[ df_indiv[ "app_user" ] == 1 ].groupby( [ "age_group" ] ).size().reset_index(name="app_users")    
        all_users  = df_indiv.groupby( [ "age_group" ] ).size().reset_index(name="all")    
        app_params = [ "app_users_fraction_0_9", "app_users_fraction_10_19",  "app_users_fraction_20_29",  
            "app_users_fraction_30_39",  "app_users_fraction_40_49", "app_users_fraction_50_59",    
            "app_users_fraction_60_69",  "app_users_fraction_70_79", "app_users_fraction_80" ]
        
        for age in constant.AGES:
            if test_params[ app_params[ age ] ] == 0 :
                users = app_users[ app_users[ "age_group"] == age ]
                np.testing.assert_equal( len( users ), 0, "nobody should have a phone in this age group" )
            else :
                n     = all_users[ all_users[ "age_group"] == age ].iloc[0,1]
                users = app_users[ app_users[ "age_group"] == age ].iloc[0,1]
                np.testing.assert_allclose( users / n, test_params[ app_params[ age ] ], atol = 0.01, err_msg = "wrong fraction of users have app in age group")
            
        df_trace     = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
        index_traced = df_trace[ ( df_trace[ "time" ] == end_time ) & ( df_trace[ "days_since_contact" ] == 0 ) ] 
        index_traced = index_traced.groupby( [ "index_ID", "traced_ID" ] ).size().reset_index(name="cons")    
        index_traced = index_traced[ index_traced[ "index_ID" ] != index_traced[ "traced_ID" ] ]
        np.testing.assert_equal( len( index_traced ) > 0, True, "no tracing has occured") 
        
        df_indiv.rename( columns = { "ID":"index_ID" }, inplace = True )
        test = pd.merge( index_traced, df_indiv, on = "index_ID", how = "left")
        np.testing.assert_equal( len( test[ test[ "app_user" ] != 1 ] ), 0, "non-app users starting tracing" ) 
        
        df_indiv.rename( columns = { "index_ID":"traced_ID" }, inplace = True )
        test = pd.merge( index_traced, df_indiv, on = "traced_ID", how = "left")
        np.testing.assert_equal( len( test[ test[ "app_user" ] != 1 ] ), 0, "non-app users being traced" )
    