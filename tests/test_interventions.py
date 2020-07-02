#!/usr/bin/env python3
"""
Tests of the individual-based model, COVID19-IBM, using the individual file

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be 
run by calling 'pytest' from project folder.  

Created: March 2020
Author: p-robot
"""

import pytest
import subprocess
import sys
import numpy as np, pandas as pd
from scipy import optimize
from scipy.stats import binom
from math import sqrt
from numpy.core.numeric import NaN
from random import randrange

sys.path.append("src/COVID19")
from parameters import ParameterSet
from model import OccupationNetworkEnum
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
                    quarantine_household_on_symptoms = 1,
                    quarantine_compliance_traced_symptoms = 1.0,
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
                    lockdown_occupation_multiplier_primary_network = 0.8,
                    lockdown_occupation_multiplier_secondary_network = 0.8,
                    lockdown_occupation_multiplier_working_network= 0.8,
                    lockdown_occupation_multiplier_retired_network= 0.8,
                    lockdown_occupation_multiplier_elderly_network= 0.8,
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
        ],
        "test_risk_score_household": [
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 500,
                    end_time = 10,
                    infectious_rate = 4,
                    self_quarantine_fraction = 1.0,
                    quarantine_household_on_symptoms = 1
                ),
                min_age_inf = 2, 
                min_age_sus = 2
            ),
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 500,
                    end_time = 10,
                    infectious_rate = 4,
                    self_quarantine_fraction = 1.0,
                    quarantine_household_on_symptoms = 1
                ),
                min_age_inf = 1, 
                min_age_sus = 3
            )
        ],
        "test_risk_score_age": [
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 500,
                    end_time = 10,
                    infectious_rate = 4,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = 1,
                    quarantine_on_traced = 1,
                    app_turn_on_time = 0
                ),
                min_age_inf = 2, 
                min_age_sus = 2
            )
        ],
        "test_risk_score_days_since_contact": [
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 500,
                    end_time = 10,
                    infectious_rate = 4,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = 1,
                    quarantine_on_traced = 1,
                    app_turn_on_time = 0
                ),
                days_since_contact = 2, 
            ),
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 500,
                    end_time = 10,
                    infectious_rate = 4,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = 1,
                    quarantine_on_traced = 1,
                    app_turn_on_time = 0
                ),
                days_since_contact = 4 
            )
        ],
        "test_risk_score_multiple_contact": [
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 500,
                    end_time = 10,
                    infectious_rate = 4,
                    self_quarantine_fraction = 1.0,
                    quarantine_compliance_traced_symptoms = 1.0,
                    quarantine_compliance_traced_positive = 1.0,
                    trace_on_symptoms = 1,
                    quarantine_on_traced = 1,
                    app_turn_on_time = 0,
                    traceable_interaction_fraction = 1,
                    daily_non_cov_symptoms_rate = 0
                ),
                days_since_contact    = 2,
                required_interactions = 1
            ),
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 500,
                    end_time = 10,
                    infectious_rate = 4,
                    self_quarantine_fraction = 1.0,
                    quarantine_compliance_traced_symptoms = 1.0,
                    quarantine_compliance_traced_positive = 1.0,
                    trace_on_symptoms = 1,
                    quarantine_on_traced = 1,
                    app_turn_on_time = 0,
                    traceable_interaction_fraction = 1,
                    daily_non_cov_symptoms_rate = 0
                ),
                days_since_contact    = 3,
                required_interactions = 2
            ),
        ],
        "test_quarantine_household_on_trace_positive_not_symptoms": [
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 100,
                    end_time = 10,
                    infectious_rate = 6,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = True,
                    trace_on_positive = True,
                    test_on_symptoms = True,
                    quarantine_on_traced = True,
                    app_turn_on_time = 0,
                    traceable_interaction_fraction = 1,
                    daily_non_cov_symptoms_rate = 0,
                    test_sensitivity = 1,
                    test_specificity = 1,
                    test_order_wait = 1,
                    test_result_wait = 1,
                    quarantine_household_on_positive = True,
                    quarantine_household_on_symptoms = True,
                    quarantine_household_on_traced_positive = True,
                    quarantine_household_on_traced_symptoms = False,
                    quarantine_dropout_traced_symptoms = 0,
                    quarantine_dropout_traced_positive = 0,
                    quarantine_compliance_traced_symptoms = 1.0,
                    quarantine_compliance_traced_positive = 1.0,
                    mean_time_to_hospital = 30
                ),
            )
        ],
        "test_traced_on_symptoms_quarantine_on_positive": [
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 100,
                    end_time = 15,
                    infectious_rate = 6,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = True,
                    trace_on_positive = True,
                    test_on_symptoms = True,
                    quarantine_on_traced = True,
                    app_turn_on_time = 1,
                    traceable_interaction_fraction = 1,
                    daily_non_cov_symptoms_rate = 0,
                    test_order_wait = 1,
                    test_result_wait = 1,
                    quarantine_household_on_positive = False,
                    quarantine_household_on_symptoms = False,
                    quarantine_household_on_traced_positive = False,
                    quarantine_household_on_traced_symptoms = False,
                    quarantine_dropout_traced_symptoms = 0,
                    quarantine_dropout_traced_positive = 0,
                    quarantine_compliance_traced_symptoms = 0.5,
                    quarantine_compliance_traced_positive = 0.9,
                    mean_time_to_hospital = 30
                ),
                tol_sd = 3
            ),
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 100,
                    end_time = 15,
                    infectious_rate = 6,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = True,
                    trace_on_positive = True,
                    test_on_symptoms = True,
                    quarantine_on_traced = True,
                    app_turn_on_time = 1,
                    traceable_interaction_fraction = 1,
                    daily_non_cov_symptoms_rate = 0,
                    test_order_wait = 1,
                    test_result_wait = 1,
                    quarantine_household_on_positive = False,
                    quarantine_household_on_symptoms = False,
                    quarantine_household_on_traced_positive = False,
                    quarantine_household_on_traced_symptoms = False,
                    quarantine_dropout_traced_symptoms = 0,
                    quarantine_dropout_traced_positive = 0,
                    quarantine_compliance_traced_symptoms = 0.0,
                    quarantine_compliance_traced_positive = 1.0,
                    mean_time_to_hospital = 30
                ),
                tol_sd = 3
            )
        ],
        "test_quarantined_have_trace_token" : [
            dict(
                test_params = dict( 
                    n_total = 20000,
                    n_seed_infection = 100,
                    end_time = 20,
                    infectious_rate = 8,
                    self_quarantine_fraction = 0.8,
                    trace_on_symptoms = True,
                    trace_on_positive = True,
                    test_on_symptoms = True,
                    quarantine_on_traced = True,
                    app_turn_on_time = 1,
                    traceable_interaction_fraction = 1,
                    daily_non_cov_symptoms_rate = 0.002,
                    test_order_wait = 1,
                    test_result_wait = 1,
                    quarantine_household_on_positive = True,
                    quarantine_household_on_symptoms = True,
                    quarantine_household_on_traced_positive = True,
                    quarantine_household_on_traced_symptoms = False,
                    quarantine_dropout_traced_symptoms = 0,
                    quarantine_dropout_traced_positive = 0,
                    quarantine_compliance_traced_symptoms = 0.5,
                    quarantine_compliance_traced_positive = 0.9,
                ),
                time_steps_test = 10
            )
        ],
        "test_priority_testing": [ 
            dict(
                test_params = dict(
                    n_total = 100000,
                    n_seed_infection = 1000,
                    end_time = 8,
                    infectious_rate = 7,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = True,
                    test_on_symptoms = True,
                    trace_on_positive = True,
                    quarantine_on_traced = 1,
                    quarantine_household_on_symptoms = 0,
                    quarantine_compliance_traced_symptoms = 0,
                    quarantine_compliance_traced_positive = 1.0,
                    quarantine_dropout_self = 0.0,
                    quarantine_dropout_traced_positive = 0.0,
                    quarantine_dropout_positive = 0.0,
                    test_order_wait  = 2,
                    test_result_wait = 1,
                    test_order_wait_priority = 0,
                    test_result_wait_priority = 1,
                    daily_non_cov_symptoms_rate = 0,
                    mean_time_to_hospital = 30,
                    traceable_interaction_fraction = 1.0,
                    quarantine_days = 7,
                    test_insensitive_period = 0,
                    test_sensitivity = 1,
                    test_specificity = 1,
                ),
                app_users_fraction    = 1.0,
                priority_test_contacts = 30
            ), 
            dict(
                test_params = dict(
                    n_total = 100000,
                    n_seed_infection = 1000,
                    end_time = 8,
                    infectious_rate = 7,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = True,
                    test_on_symptoms = True,
                    trace_on_positive = True,
                    quarantine_on_traced = 1,
                    quarantine_household_on_symptoms = 0,
                    quarantine_compliance_traced_symptoms = 0,
                    quarantine_compliance_traced_positive = 1.0,
                    quarantine_dropout_self = 0.0,
                    quarantine_dropout_traced_positive = 0.0,
                    quarantine_dropout_positive = 0.0,
                    test_order_wait  = 2,
                    test_result_wait = 1,
                    test_order_wait_priority = -1,
                    test_result_wait_priority = -1,
                    daily_non_cov_symptoms_rate = 0,
                    mean_time_to_hospital = 30,
                    traceable_interaction_fraction = 1.0,
                    quarantine_days = 7,
                    test_insensitive_period = 0,
                    test_sensitivity = 1,
                    test_specificity = 1,
                ),
                app_users_fraction    = 1.0,
                priority_test_contacts = 30
            ),
            dict(
                test_params = dict(
                    n_total = 100000,
                    n_seed_infection = 1000,
                    end_time = 8,
                    infectious_rate = 7,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = True,
                    test_on_symptoms = True,
                    trace_on_positive = True,
                    quarantine_on_traced = 1,
                    quarantine_household_on_symptoms = 0,
                    quarantine_compliance_traced_symptoms = 0,
                    quarantine_compliance_traced_positive = 1.0,
                    quarantine_dropout_self = 0.0,
                    quarantine_dropout_traced_positive = 0.0,
                    quarantine_dropout_positive = 0.0,
                    test_order_wait  = 2,
                    test_result_wait = 2,
                    test_order_wait_priority = 1,
                    test_result_wait_priority = 1,
                    daily_non_cov_symptoms_rate = 0,
                    mean_time_to_hospital = 30,
                    traceable_interaction_fraction = 1.0,
                    quarantine_days = 7,
                    test_insensitive_period = 0,
                    test_sensitivity = 1,
                    test_specificity = 1,
                ),
                app_users_fraction    = 1.0,
                priority_test_contacts = 30
            ),
        ],        
        "test_manual_trace_params" : [
            dict(
                test_params = dict( 
                    n_total = 20000,
                    n_seed_infection = 100,
                    end_time = 30,
                    infectious_rate = 12,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = 1,
                    trace_on_positive = 1,
                    test_on_symptoms = 1,
                    quarantine_on_traced = 1,
                    traceable_interaction_fraction = 1,
                    daily_non_cov_symptoms_rate = 0.002,
                    test_order_wait = 1,
                    test_result_wait = 1,
                    quarantine_household_on_positive = 1,
                    quarantine_household_on_symptoms = 1,
                    quarantine_household_on_traced_positive = 1,
                    quarantine_household_on_traced_symptoms = 0,
                    quarantine_dropout_traced_symptoms = 0,
                    quarantine_dropout_traced_positive = 0,
                    quarantine_compliance_traced_symptoms = 0.5,
                    quarantine_compliance_traced_positive = 0.9,
                    manual_trace_on = 1,
                    manual_trace_delay = 0,
                    manual_trace_time_on = 1,
                    manual_trace_on_hospitalization = 1,
                    manual_trace_on_positive = 1,
		            manual_trace_n_workers = 300,
                    manual_trace_interviews_per_worker_day = 15,
                ),
                time_steps_test = 5
            ),
            dict(
                test_params = dict(
                    n_total = 20000,
                    n_seed_infection = 100,
                    end_time = 30,
                    infectious_rate = 12,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = 1,
                    trace_on_positive = 1,
                    test_on_symptoms = 1,
                    quarantine_on_traced = 1,
                    traceable_interaction_fraction = 1,
                    daily_non_cov_symptoms_rate = 0.002,
                    test_order_wait = 1,
                    test_result_wait = 1,
                    quarantine_household_on_positive = 1,
                    quarantine_household_on_symptoms = 1,
                    quarantine_household_on_traced_positive = 1,
                    quarantine_household_on_traced_symptoms = 0,
                    quarantine_dropout_traced_symptoms = 0,
                    quarantine_dropout_traced_positive = 0,
                    quarantine_compliance_traced_symptoms = 0.5,
                    quarantine_compliance_traced_positive = 0.9,
                    manual_trace_time_on = 1,
                    manual_trace_delay = 0,
                    manual_trace_on_positive = 1,
                    manual_trace_on_hospitalization = 1,
                ),
                time_steps_test = 5
            ),
            dict(
                test_params = dict(
                    n_total = 20000,
                    n_seed_infection = 100,
                    end_time = 30,
                    infectious_rate = 12,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = 1,
                    trace_on_positive = 1,
                    test_on_symptoms = 1,
                    quarantine_on_traced = 1,
                    traceable_interaction_fraction = 0.5,
                    daily_non_cov_symptoms_rate = 0.002,
                    test_order_wait = 1,
                    test_result_wait = 1,
                    quarantine_household_on_positive = 1,
                    quarantine_household_on_symptoms = 1,
                    quarantine_household_on_traced_positive = 1,
                    quarantine_household_on_traced_symptoms = 0,
                    quarantine_dropout_traced_symptoms = 0,
                    quarantine_dropout_traced_positive = 0,
                    quarantine_compliance_traced_symptoms = 0.5,
                    quarantine_compliance_traced_positive = 0.9,
                    app_turn_on_time = 1,
                    manual_trace_time_on = 1,
                    manual_trace_delay = 0,
                    manual_trace_on_positive = 1,
                    manual_trace_on_hospitalization = 1,
                    manual_traceable_fraction_occupation = 1,
                    manual_traceable_fraction_household = 1,
                    manual_traceable_fraction_random = 1,
                ),
                time_steps_test = 5
            ),
        ],
        "test_manual_trace_only_of_given_type" : [
            dict(
                test_params = dict(
                    n_total = 20000,
                    n_seed_infection = 100,
                    end_time = 15,
                    infectious_rate = 12,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = 1,
                    trace_on_positive = 1,
                    test_on_symptoms = 1,
                    quarantine_on_traced = 1,
                    traceable_interaction_fraction = 1,
                    daily_non_cov_symptoms_rate = 0.002,
                    test_order_wait = 1,
                    test_result_wait = 1,
                    quarantine_household_on_positive = 0,
                    quarantine_household_on_symptoms = 0,
                    quarantine_household_on_traced_positive = 0,
                    quarantine_household_on_traced_symptoms = 0,
                    quarantine_dropout_traced_symptoms = 0,
                    quarantine_dropout_traced_positive = 0,
                    quarantine_compliance_traced_symptoms = 0.5,
                    quarantine_compliance_traced_positive = 0.9,
                    app_turn_on_time = 1,
                    manual_trace_delay = 0,
                    manual_trace_time_on = 1,
                    manual_trace_on_positive = 1,
                    manual_trace_on_hospitalization = 1,
                    manual_traceable_fraction_occupation = 0,
                    manual_traceable_fraction_household = 1,
                    manual_traceable_fraction_random = 0,
                ),
                time_steps_test = 5,
                interaction_type = 0,
            ),
            dict(
                test_params = dict(
                    n_total = 20000,
                    n_seed_infection = 100,
                    end_time = 20,
                    infectious_rate = 12,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = 1,
                    trace_on_positive = 1,
                    test_on_symptoms = 1,
                    quarantine_on_traced = 1,
                    traceable_interaction_fraction = 1,
                    daily_non_cov_symptoms_rate = 0.002,
                    test_order_wait = 1,
                    test_result_wait = 1,
                    quarantine_household_on_positive = 0,
                    quarantine_household_on_symptoms = 0,
                    quarantine_household_on_traced_positive = 0,
                    quarantine_household_on_traced_symptoms = 0,
                    quarantine_dropout_traced_symptoms = 0,
                    quarantine_dropout_traced_positive = 0,
                    quarantine_compliance_traced_symptoms = 0.5,
                    quarantine_compliance_traced_positive = 0.9,
                    manual_trace_time_on = 1,
                    manual_trace_delay = 0,
                    manual_trace_on_positive = 1,
                    manual_trace_on_hospitalization = 1,
                    manual_traceable_fraction_occupation = 1,
                    manual_traceable_fraction_household = 0,
                    manual_traceable_fraction_random = 0,
                    mean_work_interactions_adult = 10,
                ),
                time_steps_test = 5,
                interaction_type = 1,
            ),
            dict(
                test_params = dict(
                    n_total = 20000,
                    n_seed_infection = 100,
                    end_time = 10,
                    infectious_rate = 12,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = 1,
                    trace_on_positive = 1,
                    test_on_symptoms = 1,
                    quarantine_on_traced = 1,
                    traceable_interaction_fraction = 1,
                    daily_non_cov_symptoms_rate = 0.002,
                    test_order_wait = 1,
                    test_result_wait = 1,
                    quarantine_household_on_positive = 0,
                    quarantine_household_on_symptoms = 0,
                    quarantine_household_on_traced_positive = 0,
                    quarantine_household_on_traced_symptoms = 0,
                    quarantine_dropout_traced_symptoms = 0,
                    quarantine_dropout_traced_positive = 0,
                    quarantine_compliance_traced_symptoms = 1,
                    quarantine_compliance_traced_positive = 1,
                    manual_trace_delay = 0,
                    manual_trace_time_on = 1,
                    manual_trace_on_positive = 1,
                    manual_trace_on_hospitalization = 1,
                    manual_traceable_fraction_occupation = 0,
                    manual_traceable_fraction_household = 0,
                    manual_traceable_fraction_random = 1,
                    mean_work_interactions_child = 1,
                    mean_work_interactions_adult = 1,
                    mean_random_interactions_adult = 10,
                ),
                time_steps_test = 5,
                interaction_type = 2,
            ),
        ],
        "test_manual_trace_delay" : [
            dict(
                test_params = dict(
                    n_total = 100000,
                    n_seed_infection = 1000,
                    end_time = 12,
                    infectious_rate = 6,
                    self_quarantine_fraction = 1.0,
                    trace_on_positive = 1,
                    test_on_symptoms = 1,
                    test_on_traced = 0,
                    quarantine_on_traced = 1,
                    test_order_wait = 1,
                    test_result_wait = 1,
                    allow_clinical_diagnosis = False,
                    quarantine_dropout_traced_symptoms = 0,
                    quarantine_dropout_traced_positive = 0,
                    quarantine_compliance_traced_positive = 1,
                    quarantine_household_on_positive = 0,
                    quarantine_household_on_symptoms = 0,
                    quarantine_household_on_traced_positive = 0,
                    quarantine_household_on_traced_symptoms = 0,
                    manual_trace_time_on = 1,
                    manual_trace_delay = delay,
                    manual_trace_on_positive = 1,
                    manual_traceable_fraction_occupation = 1,
                    manual_traceable_fraction_household = 1,
                    manual_traceable_fraction_random = 1,
                    manual_trace_n_workers = 2e3,
                    manual_trace_on_hospitalization = 0
                ),
                delay = delay,
            ) for delay in [0, 1, 2, 3]
        ],
        "test_test_sensitivity": [
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 4000,
                    end_time = 12,
                    infectious_rate = 6,
                    self_quarantine_fraction = 1.0,
                    quarantine_household_on_symptoms = True,
                    test_on_symptoms = True,
                    test_on_traced = True,
                    trace_on_symptoms = True,
                    quarantine_on_traced = True,
                    test_order_wait  = 0,
                    test_result_wait  = 1,
                    app_turn_on_time = 0,
                    test_sensitivity = 0.7,
                    daily_non_cov_symptoms_rate =0.01,
                    test_specificity = 0.9,
                    test_insensitive_period = 3

                ),
            )
        ],
        "test_recursive_testing_indirect_release": [
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 4000,
                    end_time = 10,
                    infectious_rate = 6,
                    self_quarantine_fraction = 1.0,
                    trace_on_symptoms = True,
                    test_on_symptoms  = True,
                    test_on_traced    = True,
                    trace_on_positive = True,
                    quarantine_on_traced = True,
                    quarantine_household_on_positive = True,
                    quarantine_household_on_symptoms = True,
                    quarantine_household_on_traced_positive = True,
                    quarantine_household_on_traced_symptoms = False,
                    quarantine_compliance_traced_symptoms = 1.0,
                    quarantine_compliance_traced_positive = 1.0,
                    quarantine_dropout_self = 0.0,
                    quarantine_dropout_traced_positive = 0.0,
                    quarantine_dropout_positive = 0.0,
                    test_order_wait  = 1,
                    test_result_wait = 1,
                    test_specificity = 1,
                    test_insensitive_period = 0,
                    app_turn_on_time = 0,
                    test_sensitivity = 1,
                    daily_non_cov_symptoms_rate =0.00,
                ),
            )
        ],
         "test_recursive_testing": [
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 4000,
                    end_time = 10,
                    infectious_rate = 6,
                    self_quarantine_fraction = 1.0,
                    mean_time_to_hospital = 20,
                    trace_on_symptoms = True,
                    test_on_symptoms  = True,
                    test_on_traced    = True,
                    trace_on_positive = True,
                    quarantine_on_traced = True,
                    quarantine_household_on_positive = True,
                    quarantine_household_on_symptoms = True,
                    quarantine_household_on_traced_positive = False,
                    quarantine_household_on_traced_symptoms = False,
                    quarantine_compliance_traced_symptoms = 1.0,
                    quarantine_compliance_traced_positive = 1.0,
                    quarantine_dropout_self = 0.0,
                    quarantine_dropout_traced_positive = 0.0,
                    quarantine_dropout_positive = 0.0,
                    test_order_wait  = 1,
                    test_result_wait = 1,
                    test_specificity = 1,
                    test_insensitive_period = 0,
                    app_turn_on_time = 0,
                    test_sensitivity = 1,
                    allow_clinical_diagnosis = False,
                    daily_non_cov_symptoms_rate =0.00,
                ),
            )
        ],
        "test_recursive_testing_household_not_released": [
            dict(
                test_params = dict( 
                    n_total = 100000,
                    n_seed_infection = 4000,
                    end_time = 10,
                    infectious_rate = 6,
                    self_quarantine_fraction = 1.0,
                    mean_time_to_hospital = 20,
                    trace_on_symptoms = True,
                    test_on_symptoms  = True,
                    test_on_traced    = True,
                    trace_on_positive = True,
                    quarantine_on_traced = True,
                    quarantine_household_on_positive = True,
                    quarantine_household_on_symptoms = True,
                    quarantine_compliance_traced_symptoms = 1.0,
                    quarantine_compliance_traced_positive = 1.0,
                    quarantine_dropout_self = 0.0,
                    quarantine_dropout_traced_positive = 0.0,
                    quarantine_dropout_positive = 0.0,
                    test_order_wait  = 1,
                    test_result_wait = 1,
                    test_specificity = 1,
                    test_insensitive_period = 0,
                    app_turn_on_time = 0,
                    test_sensitivity = 1,
                    daily_non_cov_symptoms_rate =0.00,
                ),
            )
        ],
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
        params.set_param("test_order_wait",0)
        params.set_param("test_result_wait",0)
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
        
        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE, 
            sep = ",", comment = "#", skipinitialspace = True)
            
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE, 
            sep = ",", comment = "#", skipinitialspace = True)
        
        df_int = pd.read_csv(constant.TEST_INTERACTION_FILE, 
            comment = "#", sep = ",", skipinitialspace = True)
        
        # Merge columns from transmission file into individual file
        df_indiv = pd.merge(df_indiv, df_trans, 
            left_on = "ID", right_on = "ID_recipient", how = "left")
        
        # get the people who are in quarantine and were on the previous step
        df_quar = df_indiv[
            (df_indiv["quarantined"] == 1) & (df_indiv["time_quarantined"] < end_time)
        ]
        df_quar = df_quar.loc[:, "ID"]

        # get the number of interactions by type
        df_int = df_int.groupby(["ID_1", "type"]).size().reset_index(name="connections")

        # check to see there are no work connections
        df_test = pd.merge(
            df_quar, df_int[df_int["type"] == constant.OCCUPATION], 
            left_on = "ID", right_on = "ID_1", how="inner"
        )
        
        np.testing.assert_equal(
            len(df_test), 0, "quarantined individual with work contacts"
        )

        # check to see there are are household connections
        df_test = pd.merge(
            df_quar, df_int[df_int["type"] == constant.HOUSEHOLD], 
            left_on = "ID", right_on = "ID_1", how="inner"
        )
        np.testing.assert_equal(
            len(df_test) > 0,
            True,
            "quarantined individuals have no household connections",
        )
        
        # check to whether the number of random connections are as specified
        df_test = pd.merge(
            df_quar, df_int[df_int["type"] == constant.RANDOM], 
            left_on = "ID", right_on = "ID_1", how="inner"
        )
        
        df_test.fillna(0, inplace=True)
        
        # In some instances df_test has zero rows
        if len(df_test) == 0:
            expectation = 0
        else:
            expectation = df_test.loc[:, "connections"].mean()
        
        np.testing.assert_allclose(
            expectation,
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
        
        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_indiv = pd.merge(df_indiv, df_trans, 
            left_on = "ID", right_on = "ID_recipient", how = "left")
        
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

        # set up model
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model  = utils.get_model_swig( params )
        
        # step through time until we need to start to save the interactions each day
        for time in range( end_time ):
            model.one_time_step();             
        model.write_trace_tokens()  
        model.write_interactions_file()  
        model.write_transmissions()  

        df_int   = pd.read_csv( constant.TEST_INTERACTION_FILE, comment="#", sep=",", skipinitialspace=True )
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
        df_trans = pd.read_csv( constant.TEST_TRANSMISSION_FILE, comment="#", sep=",", skipinitialspace=True )        
        
        # get everyone who is a case or just released from hospital as these will not be traced
        df_trans = df_trans.loc[:,["ID_recipient","is_case","time_recovered","time_hospitalised"]]
        df_trans[ "not_traced"] = ( df_trans["time_recovered"] >= ( end_time - 1) ) & ( df_trans["time_hospitalised"] >= 0 ) 
        df_trans[ "not_traced"] = df_trans[ "not_traced"] | ( df_trans[ "is_case"] == 1 )  
        df_trans = df_trans[ ( df_trans[ "not_traced"] == 1 ) ]
        df_trans.rename(columns = {"ID_recipient":"traced_ID"}, inplace = True )
        
        # prepare the interaction data to get all household interations
        df_int.rename( columns = { "ID_1":"index_ID", "ID_2":"traced_ID"}, inplace = True )
        df_int[ "household" ] = ( df_int[ "house_no_1" ] == df_int[ "house_no_2" ] )
        df_int = df_int.loc[ :, [ "index_ID", "traced_ID", "household"]]

        # look at the trace token data to get all traces
        index_traced = df_trace[ ( df_trace[ "index_time" ] == end_time ) ] 
        index_traced = index_traced.groupby( [ "index_ID", "traced_ID" ] ).size().reset_index(name="cons")    
        index_traced[ "traced" ] = True

        # get all the interactions for the index cases
        index_cases = pd.DataFrame( data = { 'index_ID': index_traced.index_ID.unique() } )
        index_inter = pd.merge( index_cases, df_int, on = "index_ID", how = "left" )             
        index_inter = index_inter.groupby( [ "index_ID", "traced_ID", "household" ]).size().reset_index(name="N")    
        index_inter[ "inter" ] = True

        # check everybody with a household interaction is traced
        t = pd.merge( index_traced, index_inter, on = [ "index_ID", "traced_ID" ], how = "outer" )
        t = pd.merge( t, df_trans, on = "traced_ID", how = "left" )
        n_no_trace  = len( t[ ( t[ "traced"] != True ) & (t["household"] == True ) & (t["not_traced"] != True )] )
        n_household = len( t[ (t["household"] == True ) ] )
        np.testing.assert_equal( n_household>100, True, "insufficient household members traced to test" )
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
        for oc_net in OccupationNetworkEnum:
            expect_work = df_without.loc[ constant.OCCUPATION, ["N"] ] * test_params[ f"lockdown_occupation_multiplier{oc_net.name}" ]       
            np.testing.assert_allclose( df_with.loc[ constant.OCCUPATION, ["N"] ], expect_work, atol = sqrt( expect_work) * sd_diff, 
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

        # set up model
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model  = utils.get_model_swig( params )
        
        # step through time until we need to start to save the interactions each day
        for time in range( end_time ):
            model.one_time_step();             
        model.write_trace_tokens()  
        model.write_interactions_file()  
        model.write_individual_file()  

        df_int   = pd.read_csv( constant.TEST_INTERACTION_FILE, comment="#", sep=",", skipinitialspace=True )
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
        df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )

        # prepare the interaction data to get all interations
        df_int.rename( columns = { "ID_1":"index_ID", "ID_2":"traced_ID"}, inplace = True )
        df_int = df_int.groupby( [ "index_ID", "traced_ID" ] ).size().reset_index(name="N_cont");
                            
        # look at the trace token data to get all traces on the last day from people who met on the last day
        index_traced = df_trace[ ( df_trace[ "index_time" ] == end_time ) & ( df_trace["index_time"] == df_trace["contact_time"]) ] 
        index_traced = index_traced.groupby( [ "index_ID", "traced_ID" ] ).size().reset_index(name="cons")    
        index_traced[ "traced" ] = True
        
        # remove people in the same household since they are traced even if there is not an interaction due to hospitalisation
        house_no = df_indiv.loc[:,["ID","house_no"]]
        house_no.rename(columns={"ID":"traced_ID","house_no":"traced_house_no"},inplace=True)
        index_traced = pd.merge( index_traced, house_no, on = "traced_ID", how = "left" )
        house_no.rename(columns={"traced_ID":"index_ID","traced_house_no":"index_house_no"},inplace=True)
        index_traced = pd.merge( index_traced, house_no, on = "index_ID", how = "left" )
        index_traced = index_traced[ ( index_traced[ "traced_house_no" ] != index_traced[ "index_house_no" ])]
        
        # get all the interactions for the index cases
        index_cases = pd.DataFrame( data = { 'index_ID': index_traced.index_ID.unique() } )
        index_inter = pd.merge( index_cases, df_int, on = "index_ID", how = "left" )             
        index_inter = index_inter.groupby( [ "index_ID", "traced_ID" ]).size().reset_index(name="N")    
        index_inter[ "inter" ] = True

        # test nobody traced without an interaction
        t = pd.merge( index_traced, index_inter, on = [ "index_ID", "traced_ID" ], how = "outer" )
        no_inter = t[ t[ "inter" ] != True ] 
        np.testing.assert_equal( len(t)>100, True, "insufficient people traced to test" )    
        np.testing.assert_equal( len(no_inter), 0, "tracing someone without an interaction" )    

    
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
        
        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_indiv = pd.merge(df_indiv, df_trans, 
            left_on = "ID", right_on = "ID_recipient", how = "left")
        
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
        df_trace["days_since_contact"]=df_trace["index_time"]-df_trace["contact_time"]
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
    
    def test_risk_score_household(self, test_params, min_age_inf, min_age_sus):
        """
        Test that if risk score for household quarantining is set to 0 for the youngest
        as either the index or traced that they are not quarantined
        """
   
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        params.set_param("rng_seed", 1)      
        model  = utils.get_model_swig( params )
        
        # now update the risk scoring map
        for age_inf in range( constant.N_AGE_GROUPS ):
            for age_sus in range( constant.N_AGE_GROUPS ):
                if ( age_inf < min_age_inf ) | (age_sus < min_age_sus ):
                    model.set_risk_score_household( age_inf, age_sus, 0 )
        
        # step through the model and write the relevant files the end
        for time in range( test_params[ "end_time" ]  ):
            model.one_time_step();    
        model.write_individual_file()
        model.write_trace_tokens()
  
        df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )

        # find the index case for the new time_step
        index_traced = df_trace[ ( df_trace[ "index_time" ] == test_params[ "end_time" ] ) ] 
        index_traced = index_traced[ ( index_traced["index_ID"] != index_traced["traced_ID"] ) ]
        index_traced = index_traced.groupby( [ "index_ID", "traced_ID" ] ).size().reset_index(name="cons")    

        # get the age and house_no
        age_house        = df_indiv.loc[ :,["ID", "age_group", "house_no"]]
        index_age_house  = age_house.rename( columns = { "ID":"index_ID",  "house_no":"index_house_no",  "age_group":"index_age_group"})
        traced_age_house = age_house.rename( columns = { "ID":"traced_ID", "house_no":"traced_house_no", "age_group":"traced_age_group"})

        index_traced = pd.merge( index_traced, index_age_house,  on = "index_ID",  how = "left")
        index_traced = pd.merge( index_traced, traced_age_house, on = "traced_ID", how = "left")
            
        # now perform checks
        np.testing.assert_equal( len( index_traced ) > 50, 1, "less than 50 traced people, in-sufficient to test" )
        np.testing.assert_equal( min( index_traced[ "index_age_group"] ),  min_age_inf,  "younger people than minimum allowed are index case from which tracing has occurred" )
        np.testing.assert_equal( min( index_traced[ "traced_age_group"] ), min_age_sus, "younger people than minimum allowed are traced" )
        np.testing.assert_equal( max( index_traced[ "index_age_group"] ),  constant.N_AGE_GROUPS-1,  "oldest age group are not index cases" )
        np.testing.assert_equal( max( index_traced[ "traced_age_group"] ), constant.N_AGE_GROUPS-1,  "oldest age group are not traced" )
       
        del( model )

    def test_risk_score_age(self, test_params, min_age_inf, min_age_sus):
        """
        Test that if risk score quarantining is set to 0 for the youngest
        as either the index or traced that they are not quarantined
        """
   
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model  = utils.get_model_swig( params )
        
        # now update the risk scoring map
        for day in range( constant.MAX_DAILY_INTERACTIONS_KEPT ):
            for age_inf in range( constant.N_AGE_GROUPS ):
                for age_sus in range( constant.N_AGE_GROUPS ):
                    if ( age_inf < min_age_inf ) | (age_sus < min_age_sus ):
                        model.set_risk_score( day, age_inf, age_sus, 0 )
        
        # step through the models and write the relevant files at the end
        for time in range( test_params[ "end_time" ]  ):
            model.one_time_step();    
        model.write_individual_file()
        model.write_trace_tokens()
  
        df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )

        # find the index case for the new time_step (remove index token)
        index_traced = df_trace[ ( df_trace[ "index_time" ] == test_params[ "end_time" ]  ) ] 
        index_traced = index_traced[ ( index_traced["index_ID"] != index_traced["traced_ID"] ) ]
        index_traced = index_traced.groupby( [ "index_ID", "traced_ID" ] ).size().reset_index(name="cons")    

        # get the age and house_no
        age_house        = df_indiv.loc[ :,["ID", "age_group", "house_no"]]
        index_age_house  = age_house.rename( columns = { "ID":"index_ID",  "house_no":"index_house_no",  "age_group":"index_age_group"})
        traced_age_house = age_house.rename( columns = { "ID":"traced_ID", "house_no":"traced_house_no", "age_group":"traced_age_group"})

        index_traced = pd.merge( index_traced, index_age_house,  on = "index_ID",  how = "left")
        index_traced = pd.merge( index_traced, traced_age_house, on = "traced_ID", how = "left")
            
        # now perform checks
        np.testing.assert_equal( len( index_traced ) > 50, 1, "less than 50 traced people, in-sufficient to test" )
        np.testing.assert_equal( min( index_traced[ "index_age_group"] ),  min_age_inf,  "younger people than minimum allowed are index case from which tracing has occurred" )
        np.testing.assert_equal( min( index_traced[ "traced_age_group"] ), min_age_sus, "younger people than minimum allowed are traced" )
        np.testing.assert_equal( max( index_traced[ "index_age_group"] ),  constant.N_AGE_GROUPS-1,  "oldest age group are not index cases" )
        np.testing.assert_equal( max( index_traced[ "traced_age_group"] ), constant.N_AGE_GROUPS-1,  "oldest age group are not traced" )
       
        del( model )
    
    def test_risk_score_days_since_contact(self, test_params, days_since_contact):
        """
        Test that if risk score quarantining is set to be 0 for days greater
        than days since contact
        """
   
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model  = utils.get_model_swig( params )
        
        # now update the risk scoring map
        for day in range( constant.MAX_DAILY_INTERACTIONS_KEPT ):
            for age_inf in range( constant.N_AGE_GROUPS ):
                for age_sus in range( constant.N_AGE_GROUPS ):
                    if day > days_since_contact:
                        model.set_risk_score( day, age_inf, age_sus, 0 )
        
        # step through the model and write the trace tokens at the end
        for time in range( test_params[ "end_time" ]  ):
            model.one_time_step();    
        model.write_trace_tokens()
  
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
        df_trace["days_since_contact"]=df_trace["index_time"]-df_trace["contact_time"]


        # find the index case for the new time_step
        index_traced = df_trace[ ( df_trace[ "index_time" ] == test_params[ "end_time" ] ) ] 
        index_traced = index_traced[ ( index_traced["index_ID"] != index_traced["traced_ID"] ) ]
        index_traced = index_traced.groupby( [ "index_ID", "traced_ID", "days_since_contact" ] ).size().reset_index(name="cons")    
       
        # now perform checks
        np.testing.assert_equal( len( index_traced ) > 50, 1, "less than 50 traced people, in-sufficient to test" )
        np.testing.assert_equal( max( index_traced[ "days_since_contact"] ) <= days_since_contact, 1,  "tracing contacts from longer ago than risk score allows" )
       
        del( model )
       
    def test_risk_score_multiple_contact(self, test_params, days_since_contact, required_interactions):
        """
        Test that if risk score quarantining is set to be 0 for days greater
        than days since contact and per_interaction_score for days less or equal,
        make sure we only quarantine people who have sufficient multiple contacts
        """
        
        per_interaction_score = 1 / required_interactions + 0.0001
        
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model  = utils.get_model_swig( params )
        
        # now update the risk scoring map
        for day in range( constant.MAX_DAILY_INTERACTIONS_KEPT ):
            for age_inf in range( constant.N_AGE_GROUPS ):
                for age_sus in range( constant.N_AGE_GROUPS ):
                    model.set_risk_score( day, age_inf, age_sus, per_interaction_score )
                    if day > days_since_contact:
                        model.set_risk_score( day, age_inf, age_sus, 0 )
        
        # step through time until we need to start to save the interactions each day
        for time in range( test_params[ "end_time" ] - days_since_contact - 1 ):
            model.one_time_step();   
       
        # now get the interactions each day  
        df_inter = []
        for time in range( days_since_contact + 1 ):
            model.one_time_step();
            model.write_interactions_file();
            df_temp = pd.read_csv( constant.TEST_INTERACTION_FILE, comment="#", sep=",", skipinitialspace=True )
            df_temp[ "days_since_symptoms" ] = days_since_contact - time
            df_inter.append(df_temp)
        df_inter = pd.concat( df_inter )
        df_inter.rename( columns = { "ID_1":"index_ID", "ID_2":"traced_ID"}, inplace = True )

        # get the individuals who have the app
        model.write_individual_file()
        model.write_transmissions()
        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_indiv = pd.merge(df_indiv, df_trans, left_on = "ID", right_on = "ID_recipient", how = "left")  
        app_users = df_indiv.loc[ :,["ID", "app_user"]]
        cases     = df_indiv.loc[ :,["ID", "is_case"]]
     
        # only consider interactions between those with the app
        app_users = app_users.rename( columns = { "ID":"index_ID", "app_user":"index_app_user"})
        df_inter  = pd.merge( df_inter, app_users, on = 'index_ID', how = "left" )
        app_users = app_users.rename( columns = { "index_ID":"traced_ID", "index_app_user":"traced_app_user"})
        df_inter  = pd.merge( df_inter, app_users, on = 'traced_ID', how = "left" )
        df_inter  = df_inter[ ( df_inter[ "index_app_user" ] == True ) & ( df_inter[ "traced_app_user" ] == True ) ]
                
        # calculate the number of pairwise interactions of the relevant number of days
        df_inter = df_inter.groupby( [ "index_ID", "traced_ID" ] ).size().reset_index(name="n_interactions")
     
        # now look at the number of people asked to quarnatine
        model.write_trace_tokens()
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )

        # find the index case for the new time_step
        index_traced = df_trace[ ( df_trace[ "index_time" ] == test_params[ "end_time" ] ) ] 
        index_traced = index_traced[ ( index_traced["index_ID"] != index_traced["traced_ID"] ) ]
        index_traced = index_traced.groupby( [ "index_ID", "traced_ID" ] ).size().reset_index(name="cons")    
        index_cases  = pd.DataFrame( data = { 'index_ID': index_traced.index_ID.unique() } )

        # now for the index cases get all the individual with with they had sufficient interactions
        index_inter = pd.merge( index_cases, df_inter, on = "index_ID", how = "left" )
        index_inter = index_inter[ index_inter[ "n_interactions"] >= required_interactions]
            
        # remove interactions with index cases who will not be quarantined 
        index_inter = pd.merge( index_inter, cases, left_on = "traced_ID", right_on = "ID", how = "left" )      
        index_inter = index_inter[ (index_inter.is_case != 1)]
              
        # now perform checks
        df_all = pd.merge( index_inter, index_traced, on = [ "index_ID", "traced_ID"], how = "outer")           
        np.testing.assert_equal( len( index_traced ) > 50, 1, "less than 50 traced people, in-sufficient to test" )
        np.testing.assert_equal( len( index_inter ), len( index_traced ), "incorrect number of people traced" )
        np.testing.assert_equal( len( index_inter ), len( df_all ), "incorrect number of people traced" )
        
        del( model )

    def test_quarantine_household_on_trace_positive_not_symptoms(self, test_params ):
        """
        Test that we quarantine people's household only on a positive test
        and on symptoms we don't ask them to quarantine
        
        Note: we need to set dropout to 0 otherwise some people drop out between 
        amber and read and their households are not quarantined
        
        Also, we need to set mean time to hospital to a large number, otherwise if they were 
        hospitalised at the time the red signal went out it won't be transmitted on (however
        the household would be quarantined anyway since they are a first order contact)
        
        When testing that households are not quarantined on amber we need to set a 
        tolerance due to the fact that it is possible for 2 people in the same house
        to be direct contacts of an index case
        
        """
        
        tol = 0.01
         
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model  = utils.get_model_swig( params )
                   
        # step through time until we need to start to save the interactions each day
        for time in range( test_params[ "end_time" ] ):
            model.one_time_step();   
       
        # get the individuals who have the app
        model.write_individual_file()
        model.write_transmissions()
        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_indiv = pd.merge(df_indiv, df_trans, 
            left_on = "ID", right_on = "ID_recipient", how = "left")
                
        is_case   = df_indiv.loc[ :,["ID", "is_case", "house_no"]]
        is_case.rename( columns = { "ID":"index_ID"}, inplace = True )  
        house_no  = df_indiv.loc[ :,["ID", "house_no","is_case"]]
        house_no.rename( columns = { "ID":"traced_ID", "house_no":"traced_house_no","is_case":"is_case_traced"}, inplace = True )
      
        # remove cases from totals for house as these are not traced
        total_house = house_no[ (house_no["is_case_traced"]==0)]
        total_house = total_house.groupby( ["traced_house_no"]).size().reset_index(name="total_per_house")
      
        # now look at the number of people asked to quarantine
        model.write_trace_tokens()
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )

        # add on house_no and case status to the transmissions and count the number of traced per house
        df_trace = pd.merge( df_trace, house_no, on = "traced_ID", how = "left")
        df_trace = pd.merge( df_trace, is_case, on = "index_ID", how = "left")
        df_trace[ "same_house" ] = ( df_trace[ "house_no"] == df_trace[ "traced_house_no"] )         
        trace_grouped = df_trace.groupby( ["index_ID", "is_case","same_house","traced_house_no"]).size().reset_index(name="n_per_house")

        # find the houses where only a single person has been traced and then see if they are a case
        single_traced = trace_grouped[ (trace_grouped["n_per_house"] == 1 )]
        single_traced = pd.merge( single_traced, df_trace, on = ["index_ID", "traced_house_no"])
        single_traced = single_traced.loc[:,["index_ID", "traced_ID", "traced_house_no"]]
        single_traced.rename( columns = { "traced_ID":"ID"}, inplace = True)
        single_traced = pd.merge( single_traced, df_indiv, on = "ID" )
        single_traced.rename( columns = { "is_case":"traced_is_case"}, inplace = True)

        # for those who are not cases, we should not have traced household members 
        not_case =  trace_grouped[ ( trace_grouped[ "same_house"] == False ) & ( trace_grouped[ "is_case"] == 0 ) ]      
        np.testing.assert_equal( len( not_case ) > 50, 1, "less than 50 index cases, in-sufficient to test" )
        np.testing.assert_equal( sum( not_case[ "n_per_house"] > 1 ) / len( not_case )  < tol, 1, "traced more than one person in a household based on symptoms" )
    
        # for cases we we should have traced everyone in the household of the traded
        case = trace_grouped[ trace_grouped[ "is_case"] == 1 ]
        case = pd.merge( case, total_house, on = "traced_house_no", how = "left")
        case = case[ ( case[ "same_house"] == False ) ];
        case = pd.merge( case, single_traced, on = "traced_house_no", how = "left")
        case.fillna(0, inplace=True)
        case[ "hh_not_q"] = case[ "total_per_house"] - case[ "n_per_house"]
              
        # remove houses for only a sinlge person who is a case has been traced (we do not trace the household then)
        case = case[ (case[ "traced_is_case"]==0) ]           
      
        np.testing.assert_equal( len( case ) > 50, 1, "less than 50 index cases, in-sufficient to test" )
        np.testing.assert_equal( sum( ( case[ "hh_not_q"] > 0 ) ), 0, "member of household of first-order contact not traced on positive" )
                
        
        
    def test_traced_on_symptoms_quarantine_on_positive(self, test_params, tol_sd ):
        """
        Test that if people are sent an amber message on being traced by someone with 
        symptoms that they then quarantine when it is upgraded to a red message after 
        a positive test
        """
        symptom_time = test_params['end_time']-test_params["test_order_wait"]-test_params["test_result_wait"]

        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model  = utils.get_model_swig( params )
                
        # step through time until we need to start to save the interactions each day
        for time in range( symptom_time ):
            model.one_time_step();             
        model.write_trace_tokens()  
        model.write_individual_file()  
        df_trace_symp = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
        df_indiv_symp = pd.read_csv( constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )
        
        # now go forward and see who would have a positive test
        for time in range( test_params['end_time'] - symptom_time ):
            model.one_time_step();             
        model.write_trace_tokens()  
        model.write_individual_file()  
        df_trace_pos  = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
        df_indiv_pos = pd.read_csv( constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )       
                  
         # find everyone who has a trace token who reported on symptom_time
        df_trace_symp_t = df_trace_symp[ (df_trace_symp["index_time"] == df_trace_symp["time"] ) & (df_trace_symp["index_reason"]==0)]
        
        # remove people who have more than onn trace token (i.e they could be traced or a different reason)
        df_traced_tokens_symp = df_trace_symp.groupby("traced_ID").size().reset_index(name="n_tokens") 
        df_trace_symp_t  = pd.merge( df_trace_symp_t, df_traced_tokens_symp, how ="left", on = "traced_ID" )
        df_trace_symp_t  = df_trace_symp_t[ (df_trace_symp_t["n_tokens"]==1) ]
                                           
        # remove people who are index cases themselves (i.e they could be traced or a different reason)
        df_index_tokens_symp = df_trace_symp.groupby("index_ID").size().reset_index(name="n_per_index") 
        df_index_tokens_symp.rename(columns = {"index_ID":"traced_ID"}, inplace= True)
        df_trace_symp_t  = pd.merge( df_trace_symp_t, df_index_tokens_symp , how ="left", on = "traced_ID" )
        df_trace_symp_t  = df_trace_symp_t[ (df_trace_symp_t.n_per_index.isna())]
        
        # add the quarantine status of all the traced people 
        df_quar_symp  = df_indiv_symp.loc[:,{"ID","quarantined"}]
        df_trace_symp_t  = pd.merge( df_trace_symp_t, df_quar_symp, how ="left", left_on = "traced_ID", right_on = "ID" )
        
        # calculate the number of amber messages and number of people in quarantine because of them
        df_trace_symp_ID   = df_trace_symp_t.groupby("index_ID").size().reset_index(name="n_conn_amber")
        df_trace_symp_quar = df_trace_symp_t.loc[:,{"index_ID","quarantined"}].groupby("index_ID").sum()
        df_trace_symp_ID = pd.merge(df_trace_symp_ID, df_trace_symp_quar, on = "index_ID")
        
        # calculate the total number of people quarantined due to the amber messages and total messags
        df_trace_symp_sum = df_trace_symp_ID.sum().reset_index(name= "value")
        n_amber     = df_trace_symp_sum.loc[ df_trace_symp_sum["index"]=="n_conn_amber",{"value"}].values[0] 
        n_quar_symp = df_trace_symp_sum.loc[ df_trace_symp_sum["index"]=="quarantined",{"value"}].values[0] 
        comp_symp   = test_params["quarantine_compliance_traced_symptoms"]
        np.testing.assert_equal( n_amber > 100, True, err_msg = "Not sufficient amber messages to test")
        np.testing.assert_allclose( n_quar_symp, n_amber*comp_symp, atol=tol_sd*sqrt(n_amber*comp_symp*(1-comp_symp)), err_msg="The wrong number quarantined on an amber message")
        
        # get the list of everyone who had an amber message and did not quarantine
        df_trace_amber_nq =  df_trace_symp_t[ df_trace_symp_t["quarantined"]==0].loc[:,{"index_ID","traced_ID"}]
        
        # once a red message has been recieved check look at those who did not previously quarantine
        df_trace_pos_t  = pd.merge(df_trace_amber_nq, df_trace_pos, on = ["index_ID","traced_ID" ])
        df_trace_pos_ID = df_trace_pos_t.groupby("index_ID").size().reset_index(name="n_conn_red")
        
        # add their current quarantine status
        df_quar_pos    = df_indiv_pos.loc[:,{"ID","quarantined"}]
        df_trace_pos_t = pd.merge( df_trace_pos_t, df_quar_pos, how ="left", left_on = "traced_ID", right_on = "ID" )
        df_trace_pos_quar = df_trace_pos_t.loc[:,{"index_ID","quarantined"}].groupby("index_ID").sum()
        df_trace_pos_ID = pd.merge(df_trace_pos_ID, df_trace_pos_quar, on = "index_ID")
        
        # now calculate the totals
        df_trace_pos_sum = df_trace_pos_ID.sum().reset_index(name= "value")
        n_red       = df_trace_pos_sum.loc[ df_trace_pos_sum["index"]=="n_conn_red",{"value"}].values[0] 
        n_quar_pos  = df_trace_pos_sum.loc[ df_trace_pos_sum["index"]=="quarantined",{"value"}].values[0] 
        comp_pos  = test_params["quarantine_compliance_traced_positive"]
        np.testing.assert_equal( n_red > 100, True, err_msg = "Not sufficient red messages to non-quarantiners to test")
        np.testing.assert_allclose( n_quar_pos, n_red*comp_pos, atol=max(tol_sd*sqrt(n_red*comp_pos*(1-comp_pos)),0.5), err_msg="The wrong number quarantined on red messages")

        del( model )


    def test_quarantined_have_trace_token(self, test_params, time_steps_test ):
        """
        Test that everybody who is in quarantine has a trace token
        """
         
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model  = utils.get_model_swig( params )
                
        # step through through the initial steps to get the epidemic going
        burn_in_time = test_params[ "end_time" ] - time_steps_test
        for time in range( burn_in_time ):
            model.one_time_step();             
            
        # now record on each step those quaranatined and check for trace tokens
        for time in range( time_steps_test ):
            model.one_time_step();                       
            model.write_trace_tokens()  
            model.write_individual_file()  
       
            df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
            df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )
        
            quarantined = df_indiv[ df_indiv["quarantined"] == True ].loc[:,["ID"]]
            have_tokens = df_trace.groupby("traced_ID").size().reset_index(name="n_tokens")
            quarantined = pd.merge( quarantined, have_tokens, left_on = "ID", right_on = "traced_ID", how = "left" )
            
            np.testing.assert_equal( len( quarantined ) > 500, True, err_msg = "Not sufficient people quarantined to test")
            np.testing.assert_equal( sum( quarantined.n_tokens.isna() ), 0, err_msg = "Individuals quarantined without trace tokens")
       
        del( model )


    def test_priority_testing(self, test_params, app_users_fraction, priority_test_contacts ):
        """
        Tests that people who had the most contacts had a priority test
        
        The test needs to be run on the day the app is turned on since
        cases do not get traced (it is assumed they would ignore any message)
        and we need to make sure that traced=n unique contacts
         
        Make traced quarantine on positive perfect (i.e. always adhered to and 
        complete fidelity) and zero compliance when traced on symptoms. Therefore
        the number of people in quarantine one day after a test should be everyone
        for a priority test and only other symptomatics for non-priority tests.           
        """
        test_order_wait  = test_params[ "test_order_wait" ]
        test_result_wait = test_params[ "test_result_wait" ] 
        if test_params[ "test_order_wait_priority" ] == -1:
            test_order_wait_priority = test_params[ "test_order_wait" ]
        else :
            test_order_wait_priority = test_params[ "test_order_wait_priority" ]
        if test_params[ "test_result_wait_priority" ] == -1:
            test_result_wait_priority = test_params[ "test_result_wait" ]
        else :
            test_result_wait_priority = test_params[ "test_result_wait_priority" ]
        
        
         # define the times of the events
        test_time           = test_params[ "end_time" ] - test_result_wait
        time_non_prior_symp = test_time - test_order_wait
        time_prior_symp     = test_time - test_order_wait_priority
        time_prior_res      = test_time + test_result_wait_priority
        time_non_prior_res  = test_params[ "end_time" ]
                   
        # set up model
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        params.set_param( "app_turn_on_time", time_non_prior_symp )
        params = utils.set_app_users_fraction_all( params, app_users_fraction)
        params = utils.set_priority_test_contacts_all( params, priority_test_contacts )  
        model  = utils.get_model_swig( params )     
         
        # step through time to the time the non-priority cases are infected
        for time in range( time_non_prior_symp ):
          model.one_time_step()

        model.write_individual_file()
        non_prior_symp_df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )
        non_prior_symp_df_indiv = non_prior_symp_df_indiv.loc[:,["ID","test_status" ] ]
        non_prior_symp_df_indiv.rename( columns = { "test_status":"test_status_non_prior_symp"}, inplace = True )

        # step through time to the time the priority cases are infected
        for time in range( time_prior_symp - time_non_prior_symp ):
          model.one_time_step()

        model.write_individual_file()
        prior_symp_df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )
        prior_symp_df_indiv = prior_symp_df_indiv.loc[:,["ID","test_status" ] ]
        prior_symp_df_indiv.rename( columns = { "test_status":"test_status_prior_symp"}, inplace = True )

        # step through time the time when all the tests are taken     
        for time in range( test_time - time_prior_symp ):
          model.one_time_step()
        
        # write files
        model.write_trace_tokens()
        model.write_individual_file()
        model.write_transmissions()

        # read CSV's
        test_df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
        test_df_trans = pd.read_csv( constant.TEST_TRANSMISSION_FILE, sep = ",", comment = "#", skipinitialspace = True )
        test_df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )
        
        # determine who should get a priority test:
        # 1. get the people who developed symptoms on the last step and have the app  
        test_df_symp = test_df_trans[ ( test_df_trans["time_symptomatic"] == time_prior_symp ) | ( test_df_trans["time_symptomatic"] == time_non_prior_symp  ) ]
        test_df_symp[ "priority_symp" ] = ( test_df_trans["time_symptomatic"] ==  time_prior_symp ) 
        test_df_symp = pd.merge( test_df_symp, test_df_indiv, left_on = "ID_recipient", right_on = "ID", how = "left")
        test_df_symp = test_df_symp[ (test_df_symp["app_user"] == True ) ]
        test_df_symp = pd.merge( test_df_symp, non_prior_symp_df_indiv, left_on = "ID_recipient", right_on = "ID", how = "left")
        test_df_symp = pd.merge( test_df_symp, prior_symp_df_indiv, left_on = "ID_recipient", right_on = "ID", how = "left")
        test_df_symp = test_df_symp.loc[:,["ID_recipient","test_status","test_status_prior_symp","test_status_non_prior_symp", "priority_symp"]]
         
        # 2. get the number of contacts using the trace_tokens
        test_df_trace = test_df_trace[ ( test_df_trace["index_time"] == time_prior_symp ) | ( test_df_trace["index_time"] == time_non_prior_symp  ) ]
        test_df_trace_sum = test_df_trace.groupby("index_ID").size().reset_index(name="n_interactions")
        test_df_symp  = pd.merge( test_df_symp, test_df_trace_sum, left_on = "ID_recipient", right_on = "index_ID", how = "left" )
        test_df_symp["priority"] = ( test_df_symp["n_interactions"] > priority_test_contacts )
        if test_order_wait_priority != test_order_wait :
            test_df_symp = test_df_symp[ ( test_df_symp["priority"] == test_df_symp["priority_symp"])]

        # check we have sufficient priority and non-priority to test
        np.testing.assert_equal(sum(test_df_symp["priority"]== True) > 30, True, "Not sufficient priority cases to meaningfully test" )
        np.testing.assert_equal(sum(test_df_symp["priority"]== False) > 30, True, "Not sufficient non-priority cases to meaningfully test" )
        
        # check that all have test results are recorded at the test time
        np.testing.assert_equal(sum( (test_df_symp["test_status"]!=1)), 0, "Some test results missing" )        
       
        # check tests are ordered for non-priority people at the correct time
        if test_order_wait > 0 :
            np.testing.assert_equal(sum((test_df_symp["priority"]== False) & (test_df_symp["test_status_non_prior_symp"]!= -1)), 0, "Non-priority case not ordered on symptoms" )        

        # checks for difference in priority testing 
        if test_order_wait_priority != test_order_wait :
            # if priority tests need to wait make sure that they have the priority key
            if test_order_wait_priority > 0 :
                np.testing.assert_equal(sum((test_df_symp["priority"]== True) & (test_df_symp["test_status_prior_symp"]!= -3)), 0, "Priority case not ordered priority test" )        

            # non tests should be ordered for priority cases at the non priority time
            np.testing.assert_equal(sum((test_df_symp["priority"]== True) & (test_df_symp["test_status_non_prior_symp"]!= -2)), 0, "Priority case ordered test prior to symptoms" )        

        # filter down so we just have the trace list we are interested in
        test_df_trace = pd.merge( test_df_symp, test_df_trace, on = "index_ID", how = "left")

        # run the model x number of steps so the priority tests have come back
        for steps in range( time_prior_res - test_time ):
            model.one_time_step()
        model.write_individual_file()
        model.write_transmissions()
        model.write_trace_tokens()

        # count the number of people who are quarantined from each traced list
        df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )
        df_quar  = df_indiv.loc[:,["ID","quarantined"]]
        df_quar  = pd.merge( test_df_trace, df_indiv, left_on = "traced_ID", right_on = "ID", how = "left" )
        df_quar  = df_quar[ df_quar["quarantined"] == True ].groupby("index_ID").size().reset_index(name="n_quar")
        df_trace_sum = pd.merge( test_df_symp, df_quar, on = "index_ID", how = "left")

        priority_not_traced = df_trace_sum[ ( df_trace_sum["priority"]==True) & ( df_trace_sum["n_quar"] != df_trace_sum["n_interactions"])]
   
        np.testing.assert_equal(len(df_trace_sum[(df_trace_sum["priority"]==True)]) > 30, True, "In-sufficient priority cases to test")
        np.testing.assert_equal(len(priority_not_traced), 0, "Traced people not quarantined immediately following a priority test")

        if time_prior_res != time_non_prior_res :
            # get the people who have been uniquely traced by this index and see how many are quarantined
            df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
            df_trace = df_trace.groupby("traced_ID").size().reset_index(name="n_traced")
            df_trace = pd.merge( test_df_trace, df_trace, on = "traced_ID", how = "left" )
            df_trace = df_trace[ df_trace["n_traced"]==1]
            df_quar  = df_indiv.loc[:,["ID","quarantined"]]
            df_quar  = pd.merge( df_trace, df_indiv, left_on = "traced_ID", right_on = "ID", how = "left" )
            df_quar  = df_quar[ df_quar["quarantined"] == True ].groupby("index_ID").size().reset_index(name="n_quar_uniq")
            df_trace_sum = pd.merge( df_trace_sum, df_quar, on = "index_ID", how = "left")
            df_trace_sum.fillna(0, inplace=True)
          
            np.testing.assert_equal( sum( df_trace_sum["priority"]==False ) > 30, True, "Insufficient non-priority cases to test" )
            non_priority_traced = df_trace_sum[ ( df_trace_sum["priority"]==False) & ( df_trace_sum["n_quar_uniq"] > 1 )]

            np.testing.assert_equal( len( non_priority_traced ), 0, "Non-priority index cases traced people quarantined too soon" )
            # do a dirty measure rough measure as well to make sure not all are multiply traced
            non_priority_traced2 = df_trace_sum[ ( df_trace_sum["priority"]==False ) & ( df_trace_sum["n_interactions"] == df_trace_sum["n_quar"] ) ]
            np.testing.assert_equal( len( non_priority_traced2 ), 0, "Non-priority index cases traced people quarantined too soon (rough measure)" )
        
        # next step forward to when the non-priority test results are expected
        for time in range( time_non_prior_res - time_prior_res ):
          model.one_time_step()
        model.write_individual_file()

        # count the number of people who are quarantined from each traced list
        df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )
        df_quar  = df_indiv.loc[:,["ID","quarantined"]]
        df_quar  = pd.merge( test_df_trace, df_indiv, left_on = "traced_ID", right_on = "ID", how = "left" )
        df_quar  = df_quar[ df_quar["quarantined"] == True ].groupby("index_ID").size().reset_index(name="n_quar")
        df_trace_sum = pd.merge( test_df_symp, df_quar, on = "index_ID", how = "left")

        non_priority_not_traced = df_trace_sum[ ( df_trace_sum["priority"]==False) & ( df_trace_sum["n_quar"] != df_trace_sum["n_interactions"])]
        np.testing.assert_equal(len(df_trace_sum[(df_trace_sum["priority"]==False)]) > 30, True, "In-sufficient non-priority cases to test")
        np.testing.assert_equal(len(non_priority_not_traced), 0, "Traced people not quarantined after the longer delay of a non-prioirty test")
        
        del( model )   
        
    def test_test_sensitivity(self, test_params ):
        """
        Test that the tests results have the required sensitivity and specificity
        Make sure there sufficient true/false pos/neg and then check they 
        lie within the 99% confidence interval
        Note tests carried out of positive cases prior to the test becoming sensitive
        must be treated differently
        """
        end_time  = test_params[ "end_time" ]
        max_CI    = 0.99
        upper_CI  = ( 1 + max_CI ) / 2 
        lower_CI  = ( 1 - max_CI ) / 2 
        
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model = utils.get_model_swig( params )
        
        for time in range( end_time ):
            model.one_time_step()

        # write files
        model.write_individual_file()
        model.write_transmissions()

        # read CSV's
        df_trans = pd.read_csv( constant.TEST_TRANSMISSION_FILE, sep = ",", comment = "#", skipinitialspace = True )
        df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )
    
        # find everyone with a test result
        df_test = df_indiv.loc[:,["ID","test_status"]]
        df_test = df_test[ df_test["test_status"] >= 0 ]
        df_trans = df_trans.loc[:,["ID_recipient","time_infected", "time_symptomatic"]]
        df_test = pd.merge( df_test, df_trans, left_on = "ID", right_on = "ID_recipient", how = "left")
        df_test.fillna(-1, inplace=True)
        df_test["infected"] = (df_test["time_infected"]>-1)

        # work out whether test is sensitive based on time of infected and whether showing symptoms     
        df_test[ "test_sensitive_inf" ]  = ( ( df_test["time_infected"] != - 1 ) & ( df_test["time_infected"] <= ( end_time - test_params[ "test_insensitive_period"] ) ) )
        df_test[ "test_sensitive_symp" ] = ( ( df_test["time_symptomatic"] <= end_time ) & ( df_test["time_symptomatic"] >= 0 ) )
        df_test[ "test_sensitive" ] = ( df_test[ "test_sensitive_inf" ] | df_test[ "test_sensitive_symp" ] )
                                              
        # check the specificity of the test
        true_neg  = sum( ( df_test["infected"] == False ) & ( df_test["test_status"] == 0 ) )
        false_pos = sum( ( df_test["infected"] == False ) & ( df_test["test_status"] == 1 ) )
        p_val     = binom.cdf( true_neg, ( true_neg + false_pos ), test_params[ "test_specificity"] )
        np.testing.assert_equal( true_neg > 100, True, "In-sufficient true negatives cases to test" )
        np.testing.assert_equal( false_pos > 50, True, "In-sufficient false positives cases to test" )
        np.testing.assert_equal( p_val > lower_CI, True, "Too few false positives given the test specificity" )
        np.testing.assert_equal( p_val < upper_CI, True, "Too many false positives given the test specificity" )

        # check the sensitivity in the initial period when not sensitive
        false_neg  = sum( ( df_test["infected"] == True ) & ( df_test["test_status"] == 0 ) & ( df_test["test_sensitive"] == False ))
        true_pos   = sum( ( df_test["infected"] == True ) & ( df_test["test_status"] == 1 ) & ( df_test["test_sensitive"] == False ))
        p_val      = binom.cdf( false_neg, ( false_neg + true_pos ), test_params[ "test_specificity"] )
        np.testing.assert_equal( false_neg > 50, True, "In-sufficient false negatives in insensitive period to test" )
        np.testing.assert_equal( true_pos > 5, True, "In-sufficient true positives in insensitive period to test" )
        np.testing.assert_equal( p_val > lower_CI, True, "Too true positives in insensitive period given the test specificity" )
        np.testing.assert_equal( p_val < upper_CI, True, "Too few true positives in insensitive period the test specificity" )

        # check the sensitivity in the initial period when not sensitive
        false_neg  = sum( ( df_test["infected"] == True ) & ( df_test["test_status"] == 0 ) & ( df_test["test_sensitive"] == True ))
        true_pos   = sum( ( df_test["infected"] == True ) & ( df_test["test_status"] == 1 ) & ( df_test["test_sensitive"] == True ))
        p_val      = binom.cdf( true_pos, ( false_neg + true_pos ), test_params[ "test_sensitivity"] )
        np.testing.assert_equal( false_neg > 100, True, "In-sufficient false negatives in sensitive period to test" )
        np.testing.assert_equal( true_pos > 100, True, "In-sufficient true positives in sensitive period to test" )
        np.testing.assert_equal( p_val > lower_CI, True, "Too few true positives in sensitive period given the test sensitivity" )
        np.testing.assert_equal( p_val < upper_CI, True, "Too many true positives in sensitive period the test sensitivity" )

        del( model )
        
    def test_recursive_testing_indirect_release(self, test_params ):
        """
        Test that when recursively tested people are released that 
        those indirectly traced through them are also released
        """
        end_time  = test_params[ "end_time" ]
        symp_time = end_time - test_params[ "test_order_wait" ] - test_params[ "test_result_wait" ]

        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model = utils.get_model_swig( params )
        
        for time in range( end_time ):
            model.one_time_step()

        # write files
        model.write_trace_tokens()
      
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
        df_trace = df_trace[ df_trace[ "index_time" ] == symp_time ]
        
        # get everyone directly traced by the index case
        df_direct_trace = df_trace[ ( df_trace[ "index_ID" ] == df_trace[ "traced_from_ID" ] )]
        df_direct_trace = df_trace.loc[:,["index_ID", "traced_ID"]]
        df_direct_trace.rename( columns = {"traced_ID":"direct_traced_ID"}, inplace = True )
        
        # now get those indirectly traced (i.e. household members of traced)
        df_indirect_trace = df_trace[ ( df_trace[ "index_ID" ] != df_trace[ "traced_from_ID" ] ) & ( df_trace[ "index_ID" ] != df_trace[ "traced_ID" ] )]
        df = pd.merge( df_indirect_trace, df_direct_trace, left_on = ["index_ID", "traced_from_ID"], right_on = ["index_ID", "direct_traced_ID"], how = "left")
        df.fillna(-1, inplace=True)

        # test that everyone who has been indirectly traced, the directly traced person is still on the trace list
        np.testing.assert_equal( len( df_indirect_trace) > 500, True, "In-sufficient indirect-traced to test" )
        np.testing.assert_equal( sum( df[ "direct_traced_ID"] == -1 ), 0, "Indirect traced people where the traced from person is no longer on the trace list" )

        del( model )

    def test_manual_trace_params(self, test_params, time_steps_test ):
        """
        Tests that people are traced based on manual tracing.
        """
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )
        model  = utils.get_model_swig( params )

        # step through through the initial steps to get the epidemic going
        end_time = test_params[ "end_time" ]
        burn_in_time = test_params[ "end_time" ] - time_steps_test
        for time in range( burn_in_time ):
            model.one_time_step()

        all_pos = pd.DataFrame()

        for time in range( time_steps_test ):
            model.one_time_step()
            model.write_interactions_file()

            df_inter = pd.read_csv(constant.TEST_INTERACTION_FILE)
            all_pos = all_pos.append(df_inter[ df_inter[ "manual_traceable" ] == 1 ] )
        model.write_trace_tokens()

        np.testing.assert_equal( len( all_pos ) > 0, True, "expected manual traces do not exist" )

    def test_recursive_testing(self, test_params ):
        """
        Test checks that following a positive test for an index case we order a 
        test for all directly traced people
        
        Additionally checks that when a symptomatic index case receives a positive test
        tests are ordered for all directly traced people
        """
        end_time  = test_params[ "end_time" ]
        symp_time = end_time - test_params[ "test_order_wait" ] - test_params[ "test_result_wait" ]

        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model = utils.get_model_swig( params )
        
        for time in range( symp_time ):
            model.one_time_step()  
        
        # get the test status at the point of becoming an index case  
        model.write_trace_tokens()
        model.write_individual_file()
        
        # remove those traced by more than one index and the index cases who have been traced
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
        df_trace_uniq = df_trace.groupby("traced_ID").size().reset_index(name="n_traced")
        df_trace_uniq = df_trace_uniq[df_trace_uniq["n_traced"] == 1 ]
        df_trace = pd.merge( df_trace, df_trace_uniq, on = "traced_ID", how = "left" )
        df_trace_uniq.rename(columns={"traced_ID":"index_ID","n_traced":"n_traced_index"}, inplace = True)
        df_trace = pd.merge( df_trace, df_trace_uniq, on = "index_ID", how = "left" )
        df_trace = df_trace[ ( df_trace[ "n_traced"] == 1 ) & ( df_trace[ "n_traced_index"] == 1 )]
        
        # now get the new index case who have traced more than one and just get the direct traced
        df_trace_symp = df_trace[ ( df_trace[ "index_time" ] == symp_time ) & ( df_trace["index_reason"] == 0 ) ]
        df_trace_symp = df_trace_symp.groupby(["index_time","index_ID"]).size().reset_index(name="n_traced")
        df_trace_symp = df_trace_symp[ df_trace_symp["n_traced"] > 1 ]
        df_trace_symp["pos_at_symp"] = True
        df_trace_symp = pd.merge(df_trace_symp, df_trace, on = ["index_time","index_ID"], how = "left")
        df_trace_symp = df_trace_symp[ ( df_trace_symp["index_ID"] != df_trace_symp["traced_ID"])]
        df_trace_symp = df_trace_symp[ ( df_trace_symp["index_ID"] == df_trace_symp["traced_from_ID"])]
        df_trace_symp = df_trace_symp.loc[:,["index_time","index_ID","traced_ID",]]
                
        # go to step before symptomatic index cases get their test results back to get the status of those traced
        for time in range( end_time - symp_time - 1 ):
            model.one_time_step()

        # write files
        model.write_trace_tokens()
        model.write_individual_file()
        model.write_transmissions()
        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_indiv_1 = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_indiv_1 = pd.merge(df_indiv_1, df_trans, left_on = "ID", right_on = "ID_recipient", how = "left")

        # remove those who have been traced multiple times from the original list
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
        df_trace_uniq = df_trace.groupby("traced_ID").size().reset_index(name="n_traced_end_1")
        df_trace_symp = pd.merge( df_trace_symp, df_trace_uniq, on = "traced_ID", how = "left" )
        df_trace_symp = df_trace_symp[df_trace_symp["n_traced_end_1"] == 1 ]
   
        # now check that nobody has got a test ordered yet
        df = pd.merge( df_trace_symp, df_indiv_1, left_on = [ "traced_ID"], right_on = ["ID"], how = "left")
        np.testing.assert_equal( len( df ) > 300, True, "In-sufficient traced from index symptomatic at symptomatic time" )
        np.testing.assert_equal( sum( ( df[ "test_status"] != -2 ) ), 0, "Traced people getting a test after a symptomatic gets a positive test" )
     
        # now step forward to when the symptomatic cases get their results back
        model.one_time_step()

        # write files
        model.write_trace_tokens()
        model.write_individual_file()
        model.write_transmissions()
        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_indiv = pd.merge(df_indiv, df_trans, left_on = "ID", right_on = "ID_recipient", how = "left")

        # remove those who have been traced multiple times from the original list
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
        df_trace_uniq = df_trace.groupby("traced_ID").size().reset_index(name="n_traced_end")
        df_trace_symp = pd.merge( df_trace_symp, df_trace_uniq, on = "traced_ID", how = "left" )
        df_trace_symp = df_trace_symp[df_trace_symp["n_traced_end"] == 1 ]
   
        # all those directly traced should now be asking for a test
        df = pd.merge( df_trace_symp, df_indiv, left_on = [ "traced_ID"], right_on = ["ID"], how = "left")
        np.testing.assert_equal( len( df ) > 300, True, "In-sufficient traced from index symptomatic at symptomatic time" )
        np.testing.assert_equal( sum( ( df[ "test_status"] != -1 ) ), 0, "Traced people not getting a test after a symptomatic gets a positive test" )   
    
        # now look at at everyone who is traced directly from a positive index case and only traced once
        df_trace = pd.merge( df_trace, df_trace_uniq, on = "traced_ID", how = "left" )
        df_trace = df_trace[df_trace["n_traced_end"] == 1 ]
        df_trace = df_trace[ ( df_trace[ "index_time" ] ==  end_time ) & ( df_trace["index_reason"] == 1 ) ]
        df_trace = df_trace[ ( df_trace[ "index_ID" ] == df_trace[ "traced_from_ID" ] )]
        df_trace = df_trace[ ( df_trace[ "index_ID" ] != df_trace[ "traced_ID" ] )]
        df       = pd.merge( df_trace, df_indiv, left_on = [ "traced_ID"], right_on = ["ID"], how = "left")

        # remove traced people who were waiting for a test already
        df_indiv_1 = df_indiv_1.loc[:,["ID","test_status"]]
        df_indiv_1.rename(columns={"test_status":"test_status_1"},inplace=True)
        df = pd.merge( df, df_indiv_1, left_on = [ "traced_ID"], right_on = ["ID"], how = "left")
        df = df[ df["test_status_1" ] == -2 ]
           
        np.testing.assert_equal( len( df ) > 700, True, "In-sufficient traced from index positive at end time " )
        np.testing.assert_equal( sum( df[ "test_status" ] != -1 ), 0, "Traced people not getting a test after new positive index case" )

        del( model )

    def test_recursive_testing_household_not_released(self, test_params ):
        """
        Test that when recursively tested people that if a household 
        member of an index case tests negative they do not get released
        if the index case has tested positive
        """
        end_time   = test_params[ "end_time" ]
        index_time = end_time - 2 * ( test_params[ "test_order_wait" ] + test_params[ "test_result_wait" ] )

        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )  
        model = utils.get_model_swig( params )
        
        for time in range( index_time ):
            model.one_time_step()

        # get the symptomatic index cases        
        model.write_trace_tokens()
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
        df_symp  = df_trace[ df_trace[ "index_time"] == index_time ]
        df_symp  = df_symp[ df_symp["index_reason"] == 0].groupby( ["index_time", "index_ID"]).size().reset_index( name ="n_traced_symp")
        
        for time in range( end_time - index_time - 1 ):
            model.one_time_step()
        
        # get the test results
        model.write_individual_file()
        model.write_trace_tokens()
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_indiv = df_indiv.loc[:,["ID","test_status","house_no", "current_status", "quarantined"]]

        # first filter out all those who have been traced multiple times 
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
        df_n_trace = df_trace.groupby( "traced_ID" ).size().reset_index(name="n_trace")
        df_n_trace.rename( columns={"traced_ID":"index_ID"}, inplace = True )
        df_trace = pd.merge( df_trace, df_n_trace, on = "index_ID" )
        df_trace = df_trace[ (df_trace["n_trace"] == 1 ) ]

        # just look at those who were symptomatic when they became index cases but are now positive
        df_trace = pd.merge( df_symp, df_trace, on = [ "index_time", "index_ID"], how = "inner" )
        df_trace = df_trace[ df_trace[ "index_reason"] == 1 ]

        # next filter out those who have no susceptibles in their house
        df_house = df_indiv.loc[:,["ID","house_no"]]
        df_house.rename( columns={"ID":"index_ID", "house_no":"index_house_no"}, inplace = True )
        df_trace = pd.merge( df_trace, df_house, on = "index_ID" )
        df_trace = pd.merge( df_trace, df_indiv, left_on = "traced_ID", right_on = "ID" )
        df_trace = df_trace[ ( df_trace[ "house_no" ] == df_trace[ "index_house_no" ]) ]
        df_n_trace = df_trace.groupby( "index_ID" ).size().reset_index(name="n_traced")
        df_n_trace = df_n_trace[ (df_n_trace["n_traced"] > 1 ) ]
        df_trace   = pd.merge( df_n_trace, df_trace, on = "index_ID") 
        
        # get those with negative and positive test
        df_neg = df_trace[ df_trace["test_status"] == 0 ].loc[:,["index_ID", "traced_ID"]]
        
        # make sure that they still have the trace token on the next step
        model.one_time_step()
        model.write_trace_tokens()
        model.write_individual_file()
        df_trace = df_trace.loc[:,["index_ID", "traced_ID"]]
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
        df_trace[ "has_token"] = True
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_indiv = df_indiv.loc[:,["ID","quarantined"]]
        df_trace = pd.merge( df_trace, df_indiv, left_on = "traced_ID", right_on = "ID", how = "left")
        df_neg = pd.merge( df_neg, df_trace, on = [ "index_ID", "traced_ID"], how = "left" )
                        
        np.testing.assert_equal( len( df_neg) > 100, True, "In-sufficient household member with negative test results" )
        np.testing.assert_equal( sum( df_neg["has_token"] != True ), 0, "Household members lose their token on a negative result despite positive household member" )
        np.testing.assert_equal( sum( df_neg["quarantined"]==0), 0, "Household members released from quarantine despite positive household member" )

    def test_manual_trace_only_of_given_type(self, test_params, time_steps_test, interaction_type ):
        """
        Tests that only the given interaction_type contains manual traces.
        """
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )
        model  = utils.get_model_swig( params )

        # step through through the initial steps to get the epidemic going
        burn_in_time = test_params[ "end_time" ] - time_steps_test
        for time in range( burn_in_time ):
            model.one_time_step()

        all_pos = pd.DataFrame()

        for time in range( time_steps_test ):
            model.one_time_step()
            model.write_interactions_file()

            df_inter = pd.read_csv( constant.TEST_INTERACTION_FILE )
            all_pos = all_pos.append( df_inter[ df_inter[ "manual_traceable" ] == 1 ] )

        np.testing.assert_equal( len( all_pos[ all_pos[ "type" ] == interaction_type  ] ) > 0, True, "expected manual traces do not exist" )
        np.testing.assert_equal( len( all_pos[ all_pos[ "type" ] != interaction_type  ] ) == 0, True, "unexpected manual traces exist" )

    def test_manual_trace_delay(self, test_params, delay ):
        """
        Tests that delays in traces are accounted for by the manual tracing delay.
        
        Make sure that nobody has been manually traced from somebody who has not
        been an index for long enough (full test turn around + manual testing; exclude
        hopsitalised people who get quicker test)
        
        Check that everyone who has been an an index for the sufficient time does
        manually trace someone (if they have an interaction with a susceptible)
        """
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )
        model  = utils.get_model_swig( params )

        for time in range( test_params[ "end_time" ] ):
            model.one_time_step()
        
        # write files
        model.write_trace_tokens()
        model.write_transmissions()
        model.write_interactions_file()
        model.write_individual_file()
        df_trace = pd.read_csv( constant.TEST_TRACE_FILE, comment="#", sep=",", skipinitialspace=True )
        df_trans = pd.read_csv( constant.TEST_TRANSMISSION_FILE, sep = ",", comment = "#", skipinitialspace = True )
        df_inter = pd.read_csv(constant.TEST_INTERACTION_FILE)
        df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )

        # remove hospitalised since they get quicker tested 
        df_hosp = df_trans.loc[:,["time_hospitalised", "ID_recipient" ] ]
        df_hosp.rename( columns = { "ID_recipient":"index_ID"}, inplace = True )
        df_trace = pd.merge( df_trace, df_hosp, on = "index_ID", how = "left" )
        df_trace.fillna( 0, inplace = True )
        df_trace = df_trace[ df_trace[ "time_hospitalised"] <= 0 ]

        # remove those who have 0 interactions with non-susceptible (they will not trace anyone)
        df_inter = pd.merge( df_inter, df_indiv, left_on = "ID_2", right_on = "ID" )
        df_inter = df_inter[ df_inter[ "current_status"] == 0 ].groupby( "ID_1" ).size().reset_index( name = "n_suscept" )
        df_trace = pd.merge( df_trace, df_inter, left_on = "index_ID", right_on = "ID_1" )
        df_trace.fillna( 0, inplace = True )        
        df_trace = df_trace[ df_trace[ "n_suscept"] > 0 ]
         
        trace_delay = test_params[ "test_order_wait" ] + test_params[ "test_result_wait" ] + delay
        
        # make sure that too recent index cases have traced nobody
        df_recent = df_trace[ (df_trace[ "index_time" ] > ( test_params[ "end_time" ] - trace_delay ) ) ]
        df_recent =  df_recent.groupby( "index_ID" ).size().reset_index( name = "count" ) 
        np.testing.assert_equal( len( df_recent ) >100, True, "Insufficient index cases to test" )
        np.testing.assert_equal( sum( df_recent["count" ] != 1 ), 0, "Tracing occurred too quickly from an index case" )

        # make sure we are tracing from everybody
        df_manual_trace = df_trace[ (df_trace[ "index_time" ] == ( test_params[ "end_time" ] - trace_delay ) ) ]
        df_manual_trace = df_manual_trace.groupby( "index_ID" ).size().reset_index( name = "count" ) 
        np.testing.assert_equal( len( df_manual_trace ) >50, True, "Insufficient index cases to test" )
        np.testing.assert_equal( sum( df_manual_trace[ "count" ] == 1 ), 0, "No manual tracing occurred from index case" )

      
      