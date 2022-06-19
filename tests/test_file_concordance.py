#!/usr/bin/env python3
"""
Tests file concordance across (and within) the different filetypes of the COVID19-IBM

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be 
run by calling 'pytest' from project folder.  

Created: April 2020
Author: p-robot
"""

import os, sys
from os.path import join
import numpy as np, pandas as pd
import pytest

sys.path.append("src/COVID19")
from COVID19.model import Model, Parameters

from . import constant
from . import utilities as utils


class TestClass(object):
    """
    Test class for checking 
    """
    @pytest.mark.parametrize(
        'indiv_var, timeseries_var', [
            ('time_infected', 'total_infected'), 
            ('time_death', 'n_death'),
            ('time_recovered', 'n_recovered')]
        )
    def test_incidence_timeseries_individual(self, indiv_var, timeseries_var):
        """
        Test incidence between individual file and time series file
        """
        
        # Call the model using baseline parameters, pipe output to file, read output file
        # file_output = open(constant.TEST_OUTPUT_FILE, "w")
        # completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)

        mparams = utils.get_params_swig()
        model = utils.get_model_swig(mparams)
        model.run(verbose=False)
        df_timeseries = model.results
        model.write_individual_file()
        model.write_transmissions()
        
        # Import timeseries/transmission/individual files
        # df_timeseries = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_indiv = pd.merge(df_indiv, df_trans, 
            left_on = "ID", right_on = "ID_recipient", how = "left")
        
        incidence_indiv = df_indiv[(df_indiv[indiv_var] > 0)].groupby([indiv_var]).size().reset_index(name="connections")
        incidence_indiv.rename( columns = { indiv_var:"time"}, inplace = True )
        incidence_indiv = pd.merge( df_timeseries[ df_timeseries["time"]>1],incidence_indiv,on="time",how = "left")
        incidence_indiv.fillna(0,inplace=True) 
        incidence_indiv = incidence_indiv["connections"].values
        
        incidence_timeseries = np.diff(df_timeseries[timeseries_var].values)          
        
        np.testing.assert_array_equal(incidence_indiv, incidence_timeseries)
    
    def test_sum_to_total_infected(self):
        """
        Test that total_infected is the sum of the other compartments
        """
        
        # Call the model
        # file_output = open(constant.TEST_OUTPUT_FILE, "w")
        # completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        # df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")

        mparams = utils.get_params_swig()
        model = utils.get_model_swig(mparams)
        model.run(verbose=False)
        df_output = model.results
        
        df_sub = df_output[["n_symptoms", "n_presymptom", "n_asymptom", \
            "n_hospital", "n_death", "n_recovered", "n_critical", "n_hospitalised_recovering"]]
        
        np.testing.assert_array_equal(
            df_sub.sum(axis = 1).values, 
            df_output["total_infected"]
        )
    
    def test_columns_non_negative(self):
        """
        Test that all columns of time series file are non-negative
        """
        
        # Call the model using baseline parameters, pipe output to file, read output file
        # file_output = open(constant.TEST_OUTPUT_FILE, "w")
        # completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        # df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")

        mparams = utils.get_params_swig()
        model = utils.get_model_swig(mparams)
        model.run(verbose=False)
        df_output = model.results
        
        np.testing.assert_equal(np.all(df_output.total_infected >= 0), True)
    
    def test_infection_count(self):
        """
        Test that infection_count in individual file is consistent with the number of transmission
        events recorded in the transmission file.  
        """
        
        # Call the model
        # file_output = open(constant.TEST_OUTPUT_FILE, "w")
        # completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)

        mparams = utils.get_params_swig()
        model = utils.get_model_swig(mparams)
        model.run(verbose=False)
        model.write_individual_file()
        model.write_transmissions()
        
        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        
        np.testing.assert_equal(df_indiv.infection_count.sum(), df_trans.shape[0])

    def test_infection_count_by_age(self):
        """
        Test that infection_count in individual file is consistent with the number of transmission
        events recorded in the transmission file, when stratified by age
        """
        
        # Call the model
        # file_output = open(constant.TEST_OUTPUT_FILE, "w")
        # completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)

        mparams = utils.get_params_swig()
        model = utils.get_model_swig(mparams)
        model.run(verbose=False)
        model.write_individual_file()
        model.write_transmissions()
        
        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        # Test the counts when stratified by age
        # Sum of infection_count in individual file by age group
        infection_count_by_age_indiv = df_indiv.groupby("age_group").infection_count.sum().values
        
        # Count of infection events (rows) by age group of the recipient in transmission file
        infection_count_by_age_trans = \
            df_trans.age_group_recipient.value_counts().sort_index().values
        
        np.testing.assert_array_equal(
            infection_count_by_age_indiv,
            infection_count_by_age_trans
            )
    
    def test_quarantine_file_size(self):
        """
        Test that the quarantine files produce the same number of people in quarantine as the 
        time series file (n_quarantine column)
        """
        
        params = utils.get_params_swig()
        
        params.set_param( "end_time", 250 )
        params.set_param( "n_total", 10000 )
        params.set_param( "test_order_wait", 1 )
        params.set_param( "test_result_wait", 1 )
        params.set_param( "self_quarantine_fraction", 0.8 )
        params.set_param( "quarantine_household_on_positive", 1 )
        params.set_param( "quarantine_household_on_symptoms", 1 )
        params.set_param( "test_on_symptoms", 1 )
        params.set_param( "intervention_start_time", 42 )
        params.set_param( "app_turn_on_time", 49 )
        params.set_param( "trace_on_symptoms", 1 )
        params.set_param( "trace_on_positive", 1 )
        params.set_param( "quarantine_on_traced", 1 )

        model  = utils.get_model_swig( params )
        
        t = 1
        shapes = []; n_quarantine = []
        while t <= 200:
            model.one_time_step()
            
            # Write quarantine reasons file
            model.write_quarantine_reasons()
            n_quarantine.append(model.one_time_step_results()["n_quarantine"])
            
            
            df_quarantine_reasons = \
                pd.read_csv(constant.TEST_QUARANTINE_REASONS_FILE.substitute(T = t))
            shapes.append(df_quarantine_reasons.shape[0])
            t += 1
        
        np.testing.assert_array_equal(np.array(shapes), np.array(n_quarantine))

    def test_quarantine_totals(self):
        """
        Test the totals in the time series output are consistent with the "quarantine reasons" files
        """
        T = 100
        
        params = Parameters(constant.TEST_DATA_TEMPLATE, 1, constant.DATA_DIR_TEST,
            constant.TEST_HOUSEHOLD_TEMPLATE, constant.TEST_HOSPITAL_FILE, 1, True, True)
        
        params.set_param( "n_total", 100000 )
        
        model  = utils.get_model_swig( params )
        
        results = list()
        
        # Run for 30 days
        t = 1
        while t <= 30:
            model.one_time_step()
            model.write_quarantine_reasons()
            results.append(model.one_time_step_results())
            t += 1
        
        # Turn on self-quarantining and run for another 10 days
        model.update_running_params( "self_quarantine_fraction", 0.8 )
        model.update_running_params( "quarantine_household_on_positive", 1 )
        model.update_running_params( "quarantine_household_on_symptoms", 1 )
        
        while t <= 40:
            model.one_time_step()
            model.write_quarantine_reasons()
            results.append(model.one_time_step_results())
            t += 1
        
        # Turn on the app and run until time 100
        model.update_running_params( "app_turned_on", 1 )
        model.update_running_params( "quarantine_on_traced", 1 )
        
        while t < T:
            model.one_time_step()
            model.write_quarantine_reasons()
            results.append(model.one_time_step_results())
            t += 1
        
        df_ts = pd.DataFrame(results)
        
        # Test that quarantining events (and release events) add to total numbers in quarantine
        n_quar_events = np.cumsum(df_ts["n_quarantine_events"].values - \
            df_ts["n_quarantine_release_events"].values)
        
        np.testing.assert_array_equal(df_ts["n_quarantine"].values, n_quar_events)
        
        n_quar_events_app_user = np.cumsum(df_ts["n_quarantine_events_app_user"].values - \
            df_ts["n_quarantine_release_events_app_user"].values)
        
        np.testing.assert_array_equal(df_ts["n_quarantine_app_user"].values, n_quar_events_app_user)
        
        
        dfs =[pd.read_csv(constant.TEST_QUARANTINE_REASONS_FILE.substitute(T = t)) \
            for t in np.arange(1, T)]
        dfs = pd.concat(dfs)
        dfs["infected"] = (dfs.status > constant.EVENT_TYPES.SUSCEPTIBLE.value)*1
        dfs["recovered"] = (dfs.status == constant.EVENT_TYPES.RECOVERED.value)*1
        
        idx_vars = ["time", "app_user", "infected", "recovered"]
        dfs_gp = dfs.groupby(idx_vars)["ID"].count().reset_index()
        
        # Calculate n_quarantine
        n_quarantine = dfs_gp.groupby("time")["ID"].sum()
        n_quarantine = n_quarantine.reindex(np.arange(1, T)).fillna(0).values
        np.testing.assert_array_equal(n_quarantine, df_ts.n_quarantine.values)
        
        # Calculate n_quarantine_app_user
        n_quarantine_app_user = dfs_gp.loc[dfs_gp.app_user==1].groupby("time")["ID"].sum()
        n_quarantine_app_user = n_quarantine_app_user.reindex(np.arange(1, T)).fillna(0).values
        np.testing.assert_array_equal(n_quarantine_app_user, df_ts.n_quarantine_app_user.values)
        
        # Calculate n_quarantine_infected
        n_quarantine_infected = dfs_gp.loc[dfs_gp.infected==1].groupby("time")["ID"].sum()
        n_quarantine_infected = n_quarantine_infected.reindex(np.arange(1, T)).fillna(0).values
        np.testing.assert_array_equal(n_quarantine_infected, df_ts.n_quarantine_infected.values)
        
        # Calculate n_quarantine_recovered
        n_quarantine_recovered = dfs_gp.loc[dfs_gp.recovered==1].groupby("time")["ID"].sum()
        n_quarantine_recovered = n_quarantine_recovered.reindex(np.arange(1, T)).fillna(0).values
        
        # Calculate n_quarantine_app_user_infected
        n_quarantine_app_user_infected = \
            dfs_gp.loc[(dfs_gp.app_user==1)&(dfs_gp.infected==1)].groupby("time")["ID"].sum()
        n_quarantine_app_user_infected = \
            n_quarantine_app_user_infected.reindex(np.arange(1, T)).fillna(0).values
        np.testing.assert_array_equal(n_quarantine_app_user_infected, 
            df_ts.n_quarantine_app_user_infected.values)
        
        # Calculate n_quarantine_app_user_recovered
        n_quarantine_app_user_recovered = \
            dfs_gp.loc[(dfs_gp.app_user==1)&(dfs_gp.recovered==1)].groupby("time")["ID"].sum()
        n_quarantine_app_user_recovered = \
            n_quarantine_app_user_recovered.reindex(np.arange(1, T)).fillna(0).values
        np.testing.assert_array_equal(n_quarantine_app_user_recovered, 
            df_ts.n_quarantine_app_user_recovered.values)
