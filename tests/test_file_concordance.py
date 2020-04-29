#!/usr/bin/env python3
"""
Tests file concordance across (and within) the different filetypes of the COVID19-IBM

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be 
run by calling 'pytest' from project folder.  

Created: April 2020
Author: p-robot
"""

import subprocess, shutil, os
from os.path import join
from string import Template
import numpy as np, pandas as pd
import pytest

import sys
from scipy.signal.fir_filter_design import firwin2
sys.path.append("src/COVID19")
from parameters import ParameterSet

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
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        
        # Import timeseries/transmission/individual files
        df_timeseries = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
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
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
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
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        np.testing.assert_equal(np.all(df_output.total_infected >= 0), True)
    
    def test_infection_count(self):
        """
        Test that infection_count in individual file is consistent with the number of transmission
        events recorded in the transmission file.  
        """
        
        # Call the model
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        
        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        
        np.testing.assert_equal(df_indiv.infection_count.sum(), df_trans.shape[0])

    def test_infection_count_by_age(self):
        """
        Test that infection_count in individual file is consistent with the number of transmission
        events recorded in the transmission file, when stratified by age
        """
        
        # Call the model
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        
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
