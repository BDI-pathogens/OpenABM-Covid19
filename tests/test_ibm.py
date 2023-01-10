    #!/usr/bin/env python3
"""
Tests of the individual-based model, COVID19-IBM

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be
run by calling 'pytest' from project folder.

Created: March 2020
Author: p-robot
"""

import numpy as np, pandas as pd
import pytest

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
    def test_zero_output(self, parameter, output_column):
        """
        Set parameter value to zero should result in zero sum of output column
        """
        params = utils.get_params_swig()
        params.set_param("n_total", 50000)

        if parameter == 'fraction_asymptomatic':
            params = utils.set_fraction_asymptomatic_all( params, 0.0 )
        if parameter == 'n_seed_infection':
            params.set_param( 'n_seed_infection', 0 )
        params = utils.set_fatality_fraction_all(params, 0.0)

        # Call the model
        model  = utils.get_model_swig( params )
        
        for _ in range( params.get_param("end_time") ):
            model.one_time_step()

        df_output = model.results
        np.testing.assert_equal(df_output[output_column].sum(), 0)

    def test_zero_deaths(self):
        """
        Set fatality ratio to zero, should have no deaths if always places in the ICU
        """
        params = utils.get_params_swig()
        params.set_param("n_total", 50000)

        params = utils.set_fatality_fraction_all(params, 0.0)
        params = utils.set_location_death_icu_all(params, 1.0)
        # Call the model
        model  = utils.get_model_swig( params )
        for _ in range( params.get_param("end_time") ):
            model.one_time_step()
        
        np.testing.assert_equal(model.results["n_death"].sum(), 0)

    def test_long_time_to_death(self):
        """
        Setting mean_time_to_death beyond end of simulation should result in no recorded deaths if always place in the ICU
        """
        params = utils.get_params_swig()
        params.set_param("n_total", 50000)
        params.set_param("mean_time_to_death", 200.0)
        params.set_param("sd_time_to_death", 2.0)
        params = utils.set_location_death_icu_all(params, 1.0)

        # Call the model
        model  = utils.get_model_swig( params )
        for _ in range( params.get_param("end_time") ):
            model.one_time_step()
        
        # Call the model, pipe output to file, read output file
        np.testing.assert_equal(np.sum(model.results.n_death > 0), 0)

    def test_hospitalised_zero(self):
        """
        Test setting hospitalised fractions to zero (should be no hospitalised)
        """
        params = utils.get_params_swig()
        params.set_param("n_total", 10000)
        params = utils.set_hospitalisation_fraction_all(params, 0.0)

        # Call the model
        model  = utils.get_model_swig( params )
        for _ in range( params.get_param("end_time") ):
            model.one_time_step()

        np.testing.assert_equal(model.results["n_hospital"].sum(), 0)

    def test_fraction_asymptomatic_one(self):
        """
        Setting fraction_asymptomatic to one (should only be asymptomatics)
        """
        params = utils.get_params_swig()
        params.set_param("n_total", 10000)
        params = utils.set_fraction_asymptomatic_all( params, 1.0 )

        # Call the model
        model  = utils.get_model_swig( params )
        for _ in range( params.get_param("end_time") ):
            model.one_time_step()
        
        df_sub = model.results[["n_symptoms", "n_presymptom"]]
        np.testing.assert_equal(df_sub.sum().sum(), 0)

    def test_sum_to_total_infected(self):
        """
        Test that total_infected is the sum of the other compartments
        """
        params = utils.get_params_swig()
        params.set_param("n_total", 10000)
        
        # Call the model
        model  = utils.get_model_swig( params )
        for _ in range( params.get_param("end_time") ):
            model.one_time_step()

        df_sub = model.results[["n_symptoms", "n_presymptom", "n_asymptom", \
            "n_hospital", "n_death", "n_recovered", "n_critical", "n_hospitalised_recovering"]]

        np.testing.assert_array_equal(
            df_sub.sum(axis = 1).values,
            model.results["total_infected"]
        )

    def test_zero_recovery(self):
        """
        Setting mean_time_to_recover to be very large should avoid seeing recovered
        """
        params = utils.get_params_swig()
        params.set_param("n_total", 10000)
        params.set_param("end_time", 100)

        # Make recovery very long
        params.set_param("mean_time_to_recover", 200.0)
        params.set_param("mean_asymptomatic_to_recovery", 200.0)
        params.set_param("mean_time_hospitalised_recovery", 200.0)

        # Call the model
        model  = utils.get_model_swig( params )
        for _ in range( params.get_param("end_time") ):
            model.one_time_step()

        np.testing.assert_array_equal(
            model.results[["n_recovered"]].sum(),
            0)

    def test_zero_quarantine(self):
        """
        Test there are no individuals quarantined if all quarantine parameters are "turned off"
        """
        params = utils.get_params_swig()
        params.set_param("n_total", 10000)

        params.set_param("test_order_wait",0)
        params.set_param("test_result_wait",0)
        params = utils.turn_off_quarantine(params)

        # Call the model
        model  = utils.get_model_swig( params )
        for _ in range( params.get_param("end_time") ):
            model.one_time_step()
        
        np.testing.assert_equal(model.results["n_quarantine"].to_numpy().sum(), 0)
