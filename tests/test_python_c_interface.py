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
import pandas as pd
from random import randrange

sys.path.append("src/COVID19")
import covid19
from parameters import ParameterSet
from model import Model, Parameters, ModelParameterException, AgeGroupEnum

from . import constant
from . import utilities as utils

# STEPS > 0
STEPS = randrange(1, 10)


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
    

    def test_set_get_parameters(self):
        """
        Test the a parameter can be changed in between step runs
        """
        # Create model object
        params = Parameters(
            constant.TEST_DATA_TEMPLATE,
            constant.PARAM_LINE_NUMBER,
            constant.DATA_DIR_TEST,
            constant.TEST_HOUSEHOLD_FILE,
            constant.TEST_HOSPITAL_FILE,
            constant.HOSPITAL_PARAM_LINE_NUMBER
        )
        params.set_param( "app_users_fraction", 0.25)
        model = Model(params)

        STEPS = 2
        # Run steps
        for step in range(0, STEPS):
            model.one_time_step()
            res = model.one_time_step_results()
           
            # Try to set valid parameters
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

            model.update_running_params("quarantine_household_on_traced", 1)
            np.testing.assert_equal(
                model.get_param("quarantine_household_on_traced"), 1
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

            # Try to set/get invalid parameters
            with pytest.raises(ModelParameterException):
                model.update_running_params("wrong_parameter", 1)

            with pytest.raises(ModelParameterException):
                model.get_param("wrong_parameter")

    def test_model_total_infected_by_age(self):
        params = Parameters(
            constant.TEST_DATA_TEMPLATE,
            constant.PARAM_LINE_NUMBER,
            constant.DATA_DIR_TEST,
            constant.TEST_HOUSEHOLD_FILE,
            constant.TEST_HOSPITAL_FILE,
            constant.PARAM_LINE_NUMBER
        )
        params.set_param( "app_users_fraction", 0.25)
        model = Model(params)
        for _ in range(30):
            model.one_time_step()
        res = model.one_time_step_results()
        for age in AgeGroupEnum:
            assert res.get(f"total_infected{age.name}", None) is not None, f"Could not get total_infected{age.name}"
        assert res.get("total_infected") == sum([res.get(f"total_infected{age.name}") for age in AgeGroupEnum]), "Total infected does not equal sum of age groups"

    def test_set_lockdown_multiplier_params(self):
        params = Parameters(
            constant.TEST_DATA_TEMPLATE,
            constant.PARAM_LINE_NUMBER,
            constant.DATA_DIR_TEST,
            constant.TEST_HOUSEHOLD_FILE,
            constant.TEST_HOSPITAL_FILE,
            constant.PARAM_LINE_NUMBER
        )
        model = Model(params)
        assert covid19.get_param_lockdown_on(model.c_params) == 0

        model.update_running_params("lockdown_work_network_multiplier", 0.4)
        assert model.get_param("lockdown_work_network_multiplier") == 0.4

        model.update_running_params("lockdown_random_network_multiplier", 0.8)
        assert model.get_param("lockdown_random_network_multiplier") == 0.8

        model.update_running_params("lockdown_house_interaction_multiplier", 1.2)
        assert model.get_param("lockdown_house_interaction_multiplier") == 1.2

        model.update_running_params("lockdown_on", 1)
        assert covid19.get_param_lockdown_on(model.c_params) == 1

        model.update_running_params("lockdown_work_network_multiplier", 0.5)
        assert model.get_param("lockdown_work_network_multiplier") == 0.5

        model.update_running_params("lockdown_random_network_multiplier", 0.9)
        assert model.get_param("lockdown_random_network_multiplier") == 0.9

        model.update_running_params("lockdown_house_interaction_multiplier", 1.3)
        assert model.get_param("lockdown_house_interaction_multiplier") == 1.3

        assert covid19.get_param_lockdown_on(model.c_params) == 1
