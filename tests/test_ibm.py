    #!/usr/bin/env python3
"""
Tests of the individual-based model, COVID19-IBM

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be
run by calling 'pytest' from project folder.

Created: March 2020
Author: p-robot
"""

import subprocess
import numpy as np, pandas as pd
import pytest

import sys
sys.path.append("src/COVID19")
from parameters import ParameterSet

from . import constant
from . import utilities as utils


class TestClass(object):
    """
    Test class for checking
    """


    @pytest.mark.parametrize(
        'parameter, output_column', [
            ('n_seed_infection', 'total_infected'),
            ('fraction_asymptomatic', 'n_asymptom')]
        )
    def test_zero_output(self, parameter, output_column,tmp_path):
        """
        Set parameter value to zero should result in zero sum of output column
        """
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number = 1)
        if parameter == 'fraction_asymptomatic':
            params = utils.set_fraction_asymptomatic_all( params, 0.0 )
        if parameter == 'n_seed_infection':
            params.set_param( 'n_seed_infection', 0 )
        params = utils.set_fatality_fraction_all(params, 0.0)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Call the model, pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")

        np.testing.assert_equal(df_output[output_column].sum(), 0)


    def test_zero_deaths(self,tmp_path):
        """
        Set fatality ratio to zero, should have no deaths if always places in the ICU
        """
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number = 1)
        params = utils.set_fatality_fraction_all(params, 0.0)
        params = utils.set_location_death_icu_all(params, 1.0)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Call the model, pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")

        np.testing.assert_equal(df_output["n_death"].sum(), 0)


    def test_long_time_to_death(self,tmp_path):
        """
        Setting mean_time_to_death beyond end of simulation should result in no recorded deaths if always place in the ICU
        """
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number = 1)
        params.set_param("mean_time_to_death", 200.0)
        params.set_param("sd_time_to_death", 2.0)
        params = utils.set_location_death_icu_all(params, 1.0)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Call the model, pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")

        np.testing.assert_equal(np.sum(df_output.n_death > 0), 0)


    def test_hospitalised_zero(self,tmp_path):
        """
        Test setting hospitalised fractions to zero (should be no hospitalised)
        """
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number = 1)
        params = utils.set_hospitalisation_fraction_all(params, 0.0)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Call the model, pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")

        np.testing.assert_equal(df_output["n_hospital"].sum(), 0)


    def test_fraction_asymptomatic_one(self,tmp_path):
        """
        Setting fraction_asymptomatic to one (should only be asymptomatics)
        """
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number = 1)
        params = utils.set_fraction_asymptomatic_all( params, 1.0 )
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Call the model, pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")

        df_sub = df_output[["n_symptoms", "n_presymptom"]]


    def test_sum_to_total_infected(self,tmp_path):
        """
        Test that total_infected is the sum of the other compartments
        """

        # Call the model
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")

        df_sub = df_output[["n_symptoms", "n_presymptom", "n_asymptom", \
            "n_hospital", "n_death", "n_recovered", "n_critical", "n_hospitalised_recovering"]]

        np.testing.assert_array_equal(
            df_sub.sum(axis = 1).values,
            df_output["total_infected"]
        )


    def test_zero_recovery(self,tmp_path):
        """
        Setting mean_time_to_recover to be very large should avoid seeing recovered
        """
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number = 1)

        # Make recovery very long
        params.set_param("mean_time_to_recover", 200.0)
        params.set_param("mean_asymptomatic_to_recovery", 200.0)
        params.set_param("mean_time_hospitalised_recovery", 200.0)

        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Call the model
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")

        np.testing.assert_array_equal(
            df_output[["n_recovered"]].sum(),
            0)

    def test_mean_time_to_recover(self,tmp_path):
        """
        Setting mean_time_to_recover to be very large should avoid seeing recovered
        """
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number = 1)

        # Make recovery very long
        params.set_param("mean_time_to_recover", 200.0)
        params.set_param("mean_asymptomatic_to_recovery", 200.0)
        params.set_param("mean_time_hospitalised_recovery", 200.0)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Call the model
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")

        df_sub = df_output[["n_presymptom", "n_asymptom", "n_symptoms", \
            "n_critical", "n_hospital", "n_death", "n_hospitalised_recovering"]]

        np.testing.assert_array_equal(
            df_sub.sum(axis = 1).values,
            df_output["total_infected"].values
        )

    def test_zero_quarantine(self,tmp_path):
        """
        Test there are no individuals quarantined if all quarantine parameters are "turned off"
        """
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number = 1)
        params.set_param("test_order_wait",0)
        params.set_param("test_result_wait",0)
        params = utils.turn_off_quarantine(params)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Call the model
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
        np.testing.assert_equal(df_output["n_quarantine"].to_numpy().sum(), 0)
