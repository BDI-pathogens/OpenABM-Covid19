#!/usr/bin/env python3
"""
Tests of the individual-based model, COVID19-IBM

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

# Directories
IBM_DIR = "src"
IBM_DIR_TEST = "src_test"
DATA_DIR_TEST = "data_test"

TEST_DATA_TEMPLATE = "./tests/data/baseline_parameters.csv"
TEST_DATA_FILE = join(DATA_DIR_TEST, "test_parameters.csv")
TEST_OUTPUT_FILE = join(DATA_DIR_TEST, "test_output.csv")

TEST_HOUSEHOLD_TEMPLATE = "./tests/data/baseline_household_demographics.csv"
TEST_HOUSEHOLD_FILE = join(DATA_DIR_TEST, "test_household_demographics.csv")

NRUNS = 1

# Default parameters for tests
TEST_N_TOTAL = 10000

# Construct the executable command
EXE = "covid19ibm.exe"
command = join(IBM_DIR_TEST, EXE)

class TestClass(object):
    """
    Test class for checking 
    """
    @classmethod
    def setup_class(self):
        """
        When the class is instatiated: compile the IBM in a temporary directory
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
        
    def teardown_method(self):
        """
        
        """
        shutil.rmtree(DATA_DIR_TEST, ignore_errors = True)
    
    
    def test_columns_non_negative(self):
        """
        Test that all columns are non-negative
        """
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("n_total", TEST_N_TOTAL)
        params.write_params(TEST_DATA_FILE)
        
        # Call the model using baseline parameters, pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE, str(NRUNS), TEST_HOUSEHOLD_FILE],
            stdout = file_output)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        np.testing.assert_equal(np.all(df_output.total_infected >= 0), True)
    
    
    def test_total_infectious_rate_zero(self):
        """
        Set infectious rate to zero results in only "n_seed_infection" as total_infected
        """
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("n_total", TEST_N_TOTAL)
        params.set_param("infectious_rate", 0.0)
        params.write_params(TEST_DATA_FILE)

        # Call the model, pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE, str(NRUNS), TEST_HOUSEHOLD_FILE],
            stdout = file_output)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        output = df_output["total_infected"].iloc[-1]
        expected_output = int(params.get_param("n_seed_infection"))
        
        np.testing.assert_equal(output, expected_output)
    
    
    def test_zero_n_seed_infection(self):
        """
        Set seed cases to zero should result in no total infections
        """
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("n_total", TEST_N_TOTAL)
        params.set_param("n_seed_infection", 0)
        params.write_params(TEST_DATA_FILE)
        
        # Call the model, pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE, str(NRUNS), TEST_HOUSEHOLD_FILE],
            stdout = file_output)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        np.testing.assert_equal(df_output["total_infected"].sum(), 0)


    def test_zero_deaths(self):
        """
        Set fatality ratio to zero, should have no deaths
        """
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("fatality_fraction_0_9", 0.0)
        params.set_param("fatality_fraction_10_19", 0.0)
        params.set_param("fatality_fraction_20_29", 0.0)
        params.set_param("fatality_fraction_30_39", 0.0)
        params.set_param("fatality_fraction_40_49", 0.0)
        params.set_param("fatality_fraction_50_59", 0.0)
        params.set_param("fatality_fraction_60_69", 0.0)
        params.set_param("fatality_fraction_70_79", 0.0)
        params.set_param("fatality_fraction_80", 0.0)
        
        params.set_param("n_total", TEST_N_TOTAL)
        params.write_params(TEST_DATA_FILE)

        # Call the model, pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE, str(NRUNS), TEST_HOUSEHOLD_FILE],
            stdout = file_output)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        np.testing.assert_equal(np.sum(df_output.n_death > 0), 0)


    def test_long_time_to_death(self):
        """
        Setting mean_time_to_death beyond end of simulation should result in no recorded deaths
        """
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("mean_time_to_death", 200.0)
        params.set_param("sd_time_to_death", 2.0)
        params.set_param("end_time", 100)
        params.set_param("n_total", TEST_N_TOTAL)
        params.write_params(TEST_DATA_FILE)
        
        # Call the model, pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE, str(NRUNS), TEST_HOUSEHOLD_FILE],
            stdout = file_output)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        np.testing.assert_equal(np.sum(df_output.n_death > 0), 0)


    @pytest.mark.slow
    def test_proportion_infected(self):
        """
        Expected proportion infected, from infectious_rate (R) is: 1 - 1/R
        For R = 2.5, expected proportion infected = 1 - 1/2.5 = 0.6.  
        
        Note: this needs to be updated for many stochastic simulations
        """
        
        infectious_rate = 2.5
        proportion_infected = 1 - 1./infectious_rate
        
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("self_quarantine_fraction", 0.0)
        params.set_param("infectious_rate", infectious_rate)
        
        params.set_param("n_total", TEST_N_TOTAL)
        
        # VARYING PARAMS.  <------------------------------------------------
        params.write_params(TEST_DATA_FILE)
        
        # Call the model
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE, str(NRUNS), TEST_HOUSEHOLD_FILE],
            stdout = file_output)
        
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        np.testing.assert_almost_equal(
            df_output.total_infected.iloc[-1]/n_total, 
            0.62854
        )
    
    
    def test_mean_daily_interactions_zero(self):
        """
        Setting interactions to zero should avoid any infections,
        so number of total infections are simply the number of seed cases.  
        """
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)

        params.set_param("mean_work_interactions_child", 0)
        params.set_param("mean_work_interactions_adult", 0)
        params.set_param("mean_work_interactions_elderly", 0)

        params.set_param("mean_random_interactions_child", 0)
        params.set_param("mean_random_interactions_adult", 0)
        params.set_param("mean_random_interactions_elderly", 0)

        params.set_param("quarantined_daily_interactions", 0)
        params.set_param("hospitalised_daily_interactions", 0)

        params.set_param("n_total", TEST_N_TOTAL)
        params.write_params(TEST_DATA_FILE)

        # Call the model
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE, str(NRUNS), TEST_HOUSEHOLD_FILE],
            stdout = file_output)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",", nrows = 50)

        np.testing.assert_equal(
            df_output["total_infected"].iloc[-1], 
            int(params.get_param("n_seed_infection"))
        )


    def test_hospitalised_zero(self):
        """
        Test setting hospitalised fractions to zero (should be no hospitalised)
        """
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("n_total", TEST_N_TOTAL)
        params.set_param("hospitalised_fraction_0_9", 0.0)
        params.set_param("hospitalised_fraction_10_19", 0.0)
        params.set_param("hospitalised_fraction_20_29", 0.0)
        params.set_param("hospitalised_fraction_30_39", 0.0)
        params.set_param("hospitalised_fraction_40_49", 0.0)
        params.set_param("hospitalised_fraction_50_59", 0.0)
        params.set_param("hospitalised_fraction_60_69", 0.0)
        params.set_param("hospitalised_fraction_70_79", 0.0)
        params.set_param("hospitalised_fraction_80", 0.0)
        params.write_params(TEST_DATA_FILE)
        
        # Call the model, pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE, str(NRUNS), TEST_HOUSEHOLD_FILE],
            stdout = file_output)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        np.testing.assert_equal(
            df_output["n_hospital"].sum(), 
            0
        )
    
    def test_hospitalised_fraction_40_percent(self):
        """
        Test setting hospitalised fractions to zero (should be no hospitalised)
        """
        
        HOSPITAL_TIME_DELAY = 1
        SYMPTOMS_TIME_DELAY = 1
        
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("n_total", TEST_N_TOTAL)
        params.set_param("hospitalised_fraction_child", 0.4)
        params.set_param("hospitalised_fraction_adult", 0.4)
        params.set_param("hospitalised_fraction_elderly", 0.4)
        
        params.set_param("fraction_asymptomatic", 0.0)
        
        params.set_param("critical_fraction_child", 0.0)
        params.set_param("critical_fraction_adult", 0.0)
        params.set_param("critical_fraction_elderly", 0.0)
        
        params.set_param("mean_time_to_symptoms", float(SYMPTOMS_TIME_DELAY))
        params.set_param("sd_time_to_symptoms", 0.05)
        
        params.set_param("mean_time_to_hospital", float(HOSPITAL_TIME_DELAY))
        params.set_param("sd_time_to_hospital", 0.05)
        
        params.set_param("seasonal_flu_rate", 0.0)
        
        params.write_params(TEST_DATA_FILE)
        
        # Call the model, pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE, str(NRUNS), TEST_HOUSEHOLD_FILE],
            stdout = file_output)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        cond = (df_output["n_hospital"] != 0) & (df_output["total_infected"] > 0)
        
        print(df_output["n_hospital"][cond][:1].values)
        print(df_output["total_case"][cond].values)
        
        np.testing.assert_equal(
            np.mean(df_output["n_hospital"][cond][:1]/np.diff(df_output["total_case"][cond].values)), 
            0.4
        )

    def test_fraction_asymptomatic_zero(self):
        """
        Setting fraction_asymptomatic to zero (should be no asymptomatics)
        """
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("n_total", TEST_N_TOTAL)
        params.set_param("fraction_asymptomatic", 0.0)
        params.write_params(TEST_DATA_FILE)
        
        # Call the model, pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE, str(NRUNS), TEST_HOUSEHOLD_FILE],
            stdout = file_output)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        np.testing.assert_equal(
            df_output["n_asymptom"].sum(), 
            0
        )


    def test_fraction_asymptomatic_one(self):
        """
        Setting fraction_asymptomatic to one (should only be asymptomatics)
        """
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("n_total", TEST_N_TOTAL)
        params.set_param("fraction_asymptomatic", 1.0)
        params.write_params(TEST_DATA_FILE)
        
        # Call the model, pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE, str(NRUNS), TEST_HOUSEHOLD_FILE],
            stdout = file_output)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        df_sub = df_output[["n_symptoms", "n_presymptom"]]
        
        np.testing.assert_array_equal(
            df_sub.to_numpy().sum(), 
            0
        )


    def test_sum_to_total_infected(self):
        """
        Test that total_infected is the sum of the other compartments
        """
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("n_total", TEST_N_TOTAL)
        params.write_params(TEST_DATA_FILE)
        
        # Call the model
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE, str(NRUNS), TEST_HOUSEHOLD_FILE],
            stdout = file_output)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        df_sub = df_output[["n_symptoms", "n_presymptom", "n_asymptom", \
            "n_hospital", "n_death", "n_recovered", "n_critical"]]
        
        np.testing.assert_array_equal(
            df_sub.sum(axis = 1).values, 
            df_output["total_infected"]
        )

    
    def test_zero_recovery(self):
        """
        Setting mean_time_to_recover to be very large should avoid seeing recovered
        """
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("n_total", TEST_N_TOTAL)
        params.set_param("end_time", 150)
        
        # Make recovery very long
        params.set_param("mean_time_to_recover", 200.0)
        params.set_param("mean_asymptomatic_to_recovery", 200.0)
        
        params.write_params(TEST_DATA_FILE)
        
        # Call the model
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE, str(NRUNS), TEST_HOUSEHOLD_FILE],
            stdout = file_output)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        np.testing.assert_array_equal(
            df_output[["n_recovered"]].sum(), 
            0)
    

    def test_mean_time_to_recover(self):
        """
        Setting mean_time_to_recover to be very large should avoid seeing recovered
        """
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("n_total", TEST_N_TOTAL)
        params.set_param("end_time", 150)
        
        # Make recovery very long
        params.set_param("mean_time_to_recover", 200.0)
        params.set_param("mean_asymptomatic_to_recovery", 200.0)
        
        params.write_params(TEST_DATA_FILE)
        
        # Call the model
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE, str(NRUNS), TEST_HOUSEHOLD_FILE],
            stdout = file_output)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        df_sub = df_output[["n_presymptom", "n_asymptom", "n_symptoms", \
            "n_critical", "n_hospital", "n_death"]]
        
        np.testing.assert_array_equal(
            df_sub.sum(axis = 1).values, 
            df_output["total_infected"].values
        )


    def test_zero_quarantine(self):
        """
        No quarantine
        """
        params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
        params.set_param("n_total", TEST_N_TOTAL)
        params.set_param("end_time", 250)
        
        # Set quarantining to zero
        params.set_param("self_quarantine_fraction", 0.0)
        params.set_param("quarantine_household_on_positive", 0.0)
        params.set_param("quarantine_household_on_symptoms", 0.0)
        params.set_param("quarantine_household_on_traced", 0.0)
        params.set_param("quarantine_household_contacts_on_positive", 0.0)
        params.set_param("quarantine_on_traced", 0.0)
        params.set_param("seasonal_flu_rate", 0.0)
        
        params.write_params(TEST_DATA_FILE)
        
        # Call the model
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([command, TEST_DATA_FILE, str(NRUNS), TEST_HOUSEHOLD_FILE],
            stdout = file_output)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        np.testing.assert_equal(df_output["n_quarantine"].to_numpy().sum(), 0)


