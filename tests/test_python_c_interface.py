#!/usr/bin/env python3
"""
Very basic example to show how testing may work using pytest

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be
run by calling 'pytest' from project folder.

pytest ./tests/test_step_calc.py

Created: March 2020
Author: Daniel Montero
"""

import pytest
import sys
import numpy as np
from random import randrange, uniform

import covid19
from parameters import ParameterSet
from model import Model, Parameters, ModelParameterException, OccupationNetworkEnum, AgeGroupEnum

from . import constant
from . import utilities as utils

# STEPS > 0
STEPS = randrange(1, 10)
FLOAT_START = 1
FLOAT_END = 20


class TestClass(object):
    """
    Test class for checking
    """

    """
    def test_lockdown_on_off(self):
    """
        #Test that a lock down can be turned on and off
    """
        # Create model object
        params = Parameters(
            TEST_DATA_TEMPLATE,
            PARAM_LINE_NUMBER,
            DATA_DIR_TEST,
            TEST_DATA_HOUSEHOLD_DEMOGRAPHICS,
        )
        params.set_param( "app_users_fraction", 0.25)
        model = Model(params)

        STEPS = 100
        last  = 0
        # Run steps
        for step in range(0, STEPS):
            model.one_time_step()
            res = model.one_time_step_results()

            print( "Time =  " + str( res["time"]) +
                   "; lockdown = "  +  str( res["lockdown"]) +
                   "; app_turned_on = "  +  str( model.get_param( "app_turned_on") ) +
                   "; new_infected = " + str( res[ "total_infected" ] - last )
            )

            last = res[ "total_infected" ]

            if res["time"] == 20:
                model.update_running_params( "lockdown_on", 1 )
                np.testing.assert_equal(model.get_param("lockdown_on"), 1)


            if res["time"] == 30:
                model.update_running_params( "lockdown_on", 0 )
                model.update_running_params( "app_turned_on", 1 )
                np.testing.assert_equal(model.get_param("lockdown_on"), 0)
                np.testing.assert_equal(model.get_param("lockdown_on"), 1)

            if res["time"] == 60:
                model.update_running_params( "app_users_fraction", 0.85 )
    """


    def test_set_get_model_parameters(self,tmp_path):
        """
        Test the a parameter can be changed in between step runs
        """
        # Cr.eate model object
        params = Parameters(
            constant.TEST_DATA_TEMPLATE,
            constant.PARAM_LINE_NUMBER,
            str(tmp_path/constant.DATA_DIR_TEST),
            str(tmp_path/constant.TEST_HOUSEHOLD_FILE),
            constant.TEST_HOSPITAL_FILE,
            constant.HOSPITAL_PARAM_LINE_NUMBER
        )
        params.set_param( "n_total", 50000 )
        model = Model(params)

        # Run steps
        for step in range(0, STEPS):
            model.one_time_step()

            # Try to set valid model parameters
            model.update_running_params("test_on_symptoms", 1)
            np.testing.assert_equal(model.get_param("test_on_symptoms"), 1)

            model.update_running_params("test_on_traced", 1)
            np.testing.assert_equal(model.get_param("test_on_traced"), 1)

            model.update_running_params("quarantine_on_traced", 1)
            np.testing.assert_equal(model.get_param("quarantine_on_traced"), 1)

            model.update_running_params("traceable_interaction_fraction", 0.30)
            np.testing.assert_equal(
                model.get_param("traceable_interaction_fraction"), 0.30
            )

            model.update_running_params("tracing_network_depth", 1)
            np.testing.assert_equal(model.get_param("tracing_network_depth"), 1)

            model.update_running_params("allow_clinical_diagnosis", 1)
            np.testing.assert_equal(model.get_param("allow_clinical_diagnosis"), 1)

            model.update_running_params("quarantine_household_on_positive", 1)
            np.testing.assert_equal(
                model.get_param("quarantine_household_on_positive"), 1
            )

            model.update_running_params("quarantine_household_on_symptoms", 1)
            np.testing.assert_equal(
                model.get_param("quarantine_household_on_symptoms"), 1
            )

            model.update_running_params("quarantine_household_on_traced_positive", 1)
            np.testing.assert_equal(
                model.get_param("quarantine_household_on_traced_positive"), 1
            )

            model.update_running_params("quarantine_household_contacts_on_positive", 1)
            np.testing.assert_equal(
                model.get_param("quarantine_household_contacts_on_positive"), 1,
            )

            model.update_running_params("quarantine_days", 1)
            np.testing.assert_equal(model.get_param("quarantine_days"), 1)

            model.update_running_params("test_order_wait", 1)
            np.testing.assert_equal(model.get_param("test_order_wait"), 1)

            model.update_running_params("test_result_wait", 1)
            np.testing.assert_equal(model.get_param("test_result_wait"), 1)

            model.update_running_params("self_quarantine_fraction", 1)
            np.testing.assert_equal(model.get_param("self_quarantine_fraction"), 1)

            # Try to set/get array model parameters
            for age in AgeGroupEnum:
                contacts = randrange(1, 10)
                model.update_running_params(f"priority_test_contacts{age.name}", contacts)
                assert model.get_param(f"priority_test_contacts{age.name}") == contacts

            # Try to set/get invalid parameters
            with pytest.raises(ModelParameterException):
                model.update_running_params("wrong_parameter", 1)

            with pytest.raises(ModelParameterException):
                model.get_param("wrong_parameter")

    def test_model_total_infected_by_age(self,tmp_path):
        params = Parameters(
            constant.TEST_DATA_TEMPLATE,
            constant.PARAM_LINE_NUMBER,
            str(tmp_path/constant.DATA_DIR_TEST),
            str(tmp_path/constant.TEST_HOUSEHOLD_FILE),
            constant.TEST_HOSPITAL_FILE,
            constant.PARAM_LINE_NUMBER,
        )
        params.set_param( "n_total", 50000 )
        params.set_param( "app_users_fraction", 0.25)
        model = Model(params)
        for _ in range(30):
            model.one_time_step()
        res = model.one_time_step_results()
        for age in AgeGroupEnum:
            assert res.get(f"total_infected{age.name}", None) is not None, f"Could not get total_infected{age.name}"
        assert res.get("total_infected") == sum([res.get(f"total_infected{age.name}") for age in AgeGroupEnum]), "Total infected does not equal sum of age groups"

    def test_set_lockdown_multiplier_params(self,tmp_path):
        params = Parameters(
            constant.TEST_DATA_TEMPLATE,
            constant.PARAM_LINE_NUMBER,
            str(tmp_path/constant.DATA_DIR_TEST),
            str(tmp_path/constant.TEST_HOUSEHOLD_FILE),
            constant.TEST_HOSPITAL_FILE,
            constant.PARAM_LINE_NUMBER
        )
        params.set_param( "n_total", 50000 )

        model = Model(params)
        assert covid19.get_param_lockdown_on(model.c_params) == 0
        for oc_net in OccupationNetworkEnum:
            model.update_running_params(f"lockdown_occupation_multiplier{oc_net.name}", 0.4)
            assert model.get_param(f"lockdown_occupation_multiplier{oc_net.name}") == 0.4

        for oc_net in OccupationNetworkEnum:
            model.update_running_params(f"lockdown_occupation_multiplier{oc_net.name}", 0.8)
            assert model.get_param(f"lockdown_occupation_multiplier{oc_net.name}") == 0.8

        model.update_running_params("lockdown_house_interaction_multiplier", 1.2)
        assert model.get_param("lockdown_house_interaction_multiplier") == 1.2

        model.update_running_params("lockdown_on", 1)
        assert covid19.get_param_lockdown_on(model.c_params) == 1

        for oc_net in OccupationNetworkEnum:
            model.update_running_params(f"lockdown_occupation_multiplier{oc_net.name}", 0.5)
            assert model.get_param(f"lockdown_occupation_multiplier{oc_net.name}") == 0.5

        model.update_running_params("lockdown_random_network_multiplier", 0.9)
        assert model.get_param("lockdown_random_network_multiplier") == 0.9

        model.update_running_params("lockdown_house_interaction_multiplier", 1.3)
        assert model.get_param("lockdown_house_interaction_multiplier") == 1.3

        assert covid19.get_param_lockdown_on(model.c_params) == 1

    def test_set_get_array_parameters(self,tmp_path):
        """
        Test that an array parameter inside the C parameters structure can be changed
        """
        params = covid19.parameters()

        # Define arrays
        get_age_types = covid19.doubleArray(covid19.N_AGE_TYPES)
        set_age_types = covid19.doubleArray(covid19.N_AGE_TYPES)
        for i in range(covid19.N_AGE_TYPES):
            set_age_types[i] = uniform(FLOAT_START, FLOAT_END)

        get_work_networks = covid19.doubleArray(covid19.N_DEFAULT_OCCUPATION_NETWORKS)
        set_work_networks = covid19.doubleArray(covid19.N_DEFAULT_OCCUPATION_NETWORKS)
        for i in range(covid19.N_DEFAULT_OCCUPATION_NETWORKS):
            set_work_networks[i] = uniform(FLOAT_START, FLOAT_END)

        get_interaction_types = covid19.doubleArray(covid19.N_INTERACTION_TYPES)
        set_interaction_types = covid19.doubleArray(covid19.N_INTERACTION_TYPES)
        for i in range(covid19.N_INTERACTION_TYPES):
            set_interaction_types[i] = uniform(FLOAT_START, FLOAT_END)

        get_age_groups = covid19.doubleArray(covid19.N_AGE_GROUPS)
        set_age_groups = covid19.doubleArray(covid19.N_AGE_GROUPS)
        for i in range(covid19.N_AGE_GROUPS):
            set_age_groups[i] = uniform(FLOAT_START, FLOAT_END)

        get_household_max = covid19.doubleArray(covid19.N_HOUSEHOLD_MAX)
        set_household_max = covid19.doubleArray(covid19.N_HOUSEHOLD_MAX)
        for i in range(covid19.N_HOUSEHOLD_MAX):
            set_household_max[i] = uniform(FLOAT_START, FLOAT_END)

        # Test set/get functions
        covid19.set_param_array_mean_random_interactions(params, set_age_types)
        covid19.get_param_array_mean_random_interactions(params, get_age_types)
        for i in range(covid19.N_AGE_TYPES):
            np.testing.assert_equal(set_age_types[i], get_age_types[i])

        covid19.set_param_array_sd_random_interactions(params, set_age_types)
        covid19.get_param_array_sd_random_interactions(params, get_age_types)
        for i in range(covid19.N_AGE_TYPES):
            np.testing.assert_equal(set_age_types[i], get_age_types[i])

        covid19.set_param_array_mean_work_interactions(params, set_work_networks)
        covid19.get_param_array_mean_work_interactions(params, get_work_networks)
        for i in range(covid19.N_DEFAULT_OCCUPATION_NETWORKS):
            np.testing.assert_equal(set_work_networks[i], get_work_networks[i])

        covid19.set_param_array_relative_susceptibility(params, set_age_groups)
        covid19.get_param_array_relative_susceptibility(params, get_age_groups)
        for i in range(covid19.N_AGE_GROUPS):
            np.testing.assert_equal(set_age_groups[i], get_age_groups[i])

        covid19.set_param_array_adjusted_susceptibility(params, set_age_groups)
        covid19.get_param_array_adjusted_susceptibility(params, get_age_groups)
        for i in range(covid19.N_AGE_GROUPS):
            np.testing.assert_equal(set_age_groups[i], get_age_groups[i])

        covid19.set_param_array_relative_transmission(params, set_interaction_types)
        covid19.get_param_array_relative_transmission(params, get_interaction_types)
        for i in range(covid19.N_INTERACTION_TYPES):
            np.testing.assert_equal(set_interaction_types[i], get_interaction_types[i])

        covid19.set_param_array_hospitalised_fraction(params, set_age_groups)
        covid19.get_param_array_hospitalised_fraction(params, get_age_groups)
        for i in range(covid19.N_AGE_GROUPS):
            np.testing.assert_equal(set_age_groups[i], get_age_groups[i])

        covid19.set_param_array_critical_fraction(params, set_age_groups)
        covid19.get_param_array_critical_fraction(params, get_age_groups)
        for i in range(covid19.N_AGE_GROUPS):
            np.testing.assert_equal(set_age_groups[i], get_age_groups[i])

        covid19.set_param_array_fatality_fraction(params, set_age_groups)
        covid19.get_param_array_fatality_fraction(params, get_age_groups)
        for i in range(covid19.N_AGE_GROUPS):
            np.testing.assert_equal(set_age_groups[i], get_age_groups[i])

        covid19.set_param_array_household_size(params, set_household_max)
        covid19.get_param_array_household_size(params, get_household_max)
        for i in range(covid19.N_HOUSEHOLD_MAX):
            np.testing.assert_equal(set_household_max[i], get_household_max[i])

        covid19.set_param_array_population(params, set_age_groups)
        covid19.get_param_array_population(params, get_age_groups)
        for i in range(covid19.N_AGE_GROUPS):
            np.testing.assert_equal(set_age_groups[i], get_age_groups[i])

        covid19.set_param_array_fraction_asymptomatic(params, set_age_groups)
        covid19.get_param_array_fraction_asymptomatic(params, get_age_groups)
        for i in range(covid19.N_AGE_GROUPS):
            np.testing.assert_equal(set_age_groups[i], get_age_groups[i])

        covid19.set_param_array_mild_fraction(params, set_age_groups)
        covid19.get_param_array_mild_fraction(params, get_age_groups)
        for i in range(covid19.N_AGE_GROUPS):
            np.testing.assert_equal(set_age_groups[i], get_age_groups[i])

        covid19.set_param_array_location_death_icu(params, set_age_groups)
        covid19.get_param_array_location_death_icu(params, get_age_groups)
        for i in range(covid19.N_AGE_GROUPS):
            np.testing.assert_equal(set_age_groups[i], get_age_groups[i])

        covid19.set_param_array_app_users_fraction(params, set_age_groups)
        covid19.get_param_array_app_users_fraction(params, get_age_groups)
        for i in range(covid19.N_AGE_GROUPS):
            np.testing.assert_equal(set_age_groups[i], get_age_groups[i])


    def test_hostpital_admissions(self,tmp_path):
        params = Parameters(
            constant.TEST_DATA_TEMPLATE,
            constant.PARAM_LINE_NUMBER,
            str(tmp_path/constant.DATA_DIR_TEST),
            str(tmp_path/constant.TEST_HOUSEHOLD_FILE),
            constant.TEST_HOSPITAL_FILE,
)
        params.set_param( "n_total", 50000 )
        model = Model(params)
        daily_hospitalisations = []
        for step in range(50):
            model.one_time_step()
            daily_h = model.one_time_step_results()["hospital_admissions"]
            daily_hospitalisations.append(daily_h)
            assert sum(daily_hospitalisations) == model.one_time_step_results()["hospital_admissions_total"]
        assert sum(daily_hospitalisations) > 0

    def test_icu_entry(self,tmp_path):
        params = Parameters(
            constant.TEST_DATA_TEMPLATE,
            constant.PARAM_LINE_NUMBER,
            str(tmp_path/constant.DATA_DIR_TEST),
            str(tmp_path/constant.TEST_HOUSEHOLD_FILE),
            constant.TEST_HOSPITAL_FILE,
)
        params.set_param( "n_total", 50000 )
        model = Model(params)
        daily_critical = []
        for step in range(50):
            model.one_time_step()
            daily_c = model.one_time_step_results()["hospital_to_critical_daily"]
            daily_critical.append(daily_c)
            assert sum(daily_critical) == model.one_time_step_results()["hospital_to_critical_total"]
        assert sum(daily_critical) > 0

    def test_deaths(self,tmp_path):
        params = Parameters(
            constant.TEST_DATA_TEMPLATE,
            constant.PARAM_LINE_NUMBER,
            str(tmp_path/constant.DATA_DIR_TEST),
            str(tmp_path/constant.TEST_HOUSEHOLD_FILE),
            constant.TEST_HOSPITAL_FILE,
)
        params.set_param( "n_total", 50000);
        model = Model(params)
        daily_deaths = []
        for step in range(50):
            model.one_time_step()
            daily_death = model.one_time_step_results()["daily_death"]
            daily_deaths.append(daily_death)
            assert sum(daily_deaths) == model.one_time_step_results()["total_death"]
        assert sum(daily_deaths) > 0

    def test_daily_deaths_by_age(self,tmp_path):
        params = Parameters(
            constant.TEST_DATA_TEMPLATE,
            constant.PARAM_LINE_NUMBER,
            str(tmp_path/constant.DATA_DIR_TEST),
            str(tmp_path/constant.TEST_HOUSEHOLD_FILE),
            constant.TEST_HOSPITAL_FILE,
)
        params.set_param( "n_total", 50000 )
        model = Model(params)
                
        for step in range(50):
            model.one_time_step()
            daily_death = model.one_time_step_results()["daily_death"]

            sum_daily_deaths_by_age = 0
            for age in AgeGroupEnum:
                sum_daily_deaths_by_age += model.one_time_step_results()[f"daily_death{age.name}"]
            assert  daily_death == sum_daily_deaths_by_age
        assert model.one_time_step_results()[f"daily_death{age.name}"] > 0



    def test_update_fatality_fraction(self,tmp_path):
        params = Parameters(
            constant.TEST_DATA_TEMPLATE,
            constant.PARAM_LINE_NUMBER,
            str(tmp_path/constant.DATA_DIR_TEST),
            str(tmp_path/constant.TEST_HOUSEHOLD_FILE),
            constant.TEST_HOSPITAL_FILE,
)
        params.set_param( "n_total", 50000 )
        model = Model(params)
        assert model.get_param("fatality_fraction_80") == 1.0

        model.update_running_params("fatality_fraction_80", 0.6)
        assert model.get_param("fatality_fraction_80") == 0.6



