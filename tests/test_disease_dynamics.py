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
import sys
import numpy as np, pandas as pd
from math import sqrt

sys.path.append("src/COVID19")

from parameters import ParameterSet
from model import OccupationNetworkEnum, VaccineTypesEnum, VaccineStatusEnum, VaccineSchedule, EVENT_TYPES, AgeGroupEnum
from . import constant
from . import utilities as utils
import covid19
import COVID19.model as abm

def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )

class TestClass(object):
    params = {
        "test_total_infectious_rate_zero": [dict()],
        "test_zero_infected": [dict()],
        "test_zero_recovery": [dict()],
        "test_zero_deaths": [dict()],
        "test_disease_transition_times": [
            dict(
                test_params = dict(
                    mean_time_to_symptoms=4.0,
                    sd_time_to_symptoms=2.0,
                    mean_time_to_hospital=1.0,
                    mean_time_to_critical=1.0,
                    mean_time_to_recover=20.0,
                    sd_time_to_recover=8.0,
                    mean_time_to_death=12.0,
                    sd_time_to_death=5.0,
                    mean_asymptomatic_to_recovery=15.0,
                    sd_asymptomatic_to_recovery=5.0,
                    mean_time_hospitalised_recovery=6,
                    sd_time_hospitalised_recovery=3,
                    mean_time_critical_survive=4,
                    sd_time_critical_survive=2,
                )
            ),
            dict(
                test_params = dict(
                    mean_time_to_symptoms=4.5,
                    sd_time_to_symptoms=2.0,
                    mean_time_to_hospital=1.2,
                    mean_time_to_critical=1.2,
                    mean_time_to_recover=18.0,
                    sd_time_to_recover=7.0,
                    mean_time_to_death=14.0,
                    sd_time_to_death=5.5,
                    mean_asymptomatic_to_recovery=17.0,
                    sd_asymptomatic_to_recovery=5.0,
                    mean_time_hospitalised_recovery=8,
                    sd_time_hospitalised_recovery=4,
                    mean_time_critical_survive=5,
                    sd_time_critical_survive=2,
                )
            ),
            dict(
                test_params = dict(
                    mean_time_to_symptoms=5.0,
                    sd_time_to_symptoms=2.5,
                    mean_time_to_hospital=1.4,
                    mean_time_to_critical=1.4,
                    mean_time_to_recover=16.0,
                    sd_time_to_recover=7.0,
                    mean_time_to_death=16.0,
                    sd_time_to_death=6,
                    mean_asymptomatic_to_recovery=18.0,
                    sd_asymptomatic_to_recovery=7.0,
                    mean_time_hospitalised_recovery=10,
                    sd_time_hospitalised_recovery=5,
                    mean_time_critical_survive=6,
                    sd_time_critical_survive=2,
                )
            ),
            dict(
                test_params = dict(
                    mean_time_to_symptoms=5.5,
                    sd_time_to_symptoms=2.5,
                    mean_time_to_hospital=1.6,
                    mean_time_to_critical=1.6,
                    mean_time_to_recover=14.0,
                    sd_time_to_recover=6.0,
                    mean_time_to_death=17.0,
                    sd_time_to_death=5,
                    mean_asymptomatic_to_recovery=12.0,
                    sd_asymptomatic_to_recovery=4.0,
                    mean_time_hospitalised_recovery=12,
                    sd_time_hospitalised_recovery=6,
                    mean_time_critical_survive=3,
                    sd_time_critical_survive=2,
                )
            ),
            dict(
                test_params = dict(
                    mean_time_to_symptoms=6.0,
                    sd_time_to_symptoms=3.0,
                    mean_time_to_hospital=1.8,
                    mean_time_to_critical=2.0,
                    mean_time_to_recover=12.0,
                    sd_time_to_recover=6.0,
                    mean_time_to_death=18.0,
                    sd_time_to_death=6,
                    mean_asymptomatic_to_recovery=14.0,
                    sd_asymptomatic_to_recovery=5.0,
                    mean_time_hospitalised_recovery=8,
                    sd_time_hospitalised_recovery=3,
                    mean_time_critical_survive=6,
                    sd_time_critical_survive=3,
                )
            ),
        ],
        "test_disease_outcome_proportions": [
            dict(
                test_params = dict(
                    fraction_asymptomatic_0_9   = 0.05,
                    fraction_asymptomatic_10_19 = 0.05,
                    fraction_asymptomatic_20_29 = 0.05,
                    fraction_asymptomatic_30_39 = 0.05,
                    fraction_asymptomatic_40_49 = 0.05,
                    fraction_asymptomatic_50_59 = 0.05,
                    fraction_asymptomatic_60_69 = 0.05,
                    fraction_asymptomatic_70_79 = 0.05,
                    fraction_asymptomatic_80    = 0.05,
                    mild_fraction_0_9           = 0.00,
                    mild_fraction_10_19         = 0.00,
                    mild_fraction_20_29         = 0.00,
                    mild_fraction_30_39         = 0.00,
                    mild_fraction_40_49         = 0.00,
                    mild_fraction_50_59         = 0.00,
                    mild_fraction_60_69         = 0.00,
                    mild_fraction_70_79         = 0.00,
                    mild_fraction_80            = 0.00,
                    hospitalised_fraction_0_9  =0.05,
                    hospitalised_fraction_10_19=0.10,
                    hospitalised_fraction_20_29=0.10,
                    hospitalised_fraction_30_39=0.20,
                    hospitalised_fraction_40_49=0.20,
                    hospitalised_fraction_50_59=0.20,
                    hospitalised_fraction_60_69=0.30,
                    hospitalised_fraction_70_79=0.30,
                    hospitalised_fraction_80   =0.50,
                    critical_fraction_0_9  =0.05,
                    critical_fraction_10_19=0.10,
                    critical_fraction_20_29=0.10,
                    critical_fraction_30_39=0.20,
                    critical_fraction_40_49=0.20,
                    critical_fraction_50_59=0.20,
                    critical_fraction_60_69=0.30,
                    critical_fraction_70_79=0.30,
                    critical_fraction_80   =0.50,
                    fatality_fraction_0_9  =0.05,
                    fatality_fraction_10_19=0.10,
                    fatality_fraction_20_29=0.10,
                    fatality_fraction_30_39=0.20,
                    fatality_fraction_40_49=0.20,
                    fatality_fraction_50_59=0.20,
                    fatality_fraction_60_69=0.30,
                    fatality_fraction_70_79=0.30,
                    fatality_fraction_80   =0.50,
                )
            ),
            dict(
                test_params = dict(
                    fraction_asymptomatic_0_9   = 0.15,
                    fraction_asymptomatic_10_19 = 0.15,
                    fraction_asymptomatic_20_29 = 0.15,
                    fraction_asymptomatic_30_39 = 0.15,
                    fraction_asymptomatic_40_49 = 0.15,
                    fraction_asymptomatic_50_59 = 0.15,
                    fraction_asymptomatic_60_69 = 0.15,
                    fraction_asymptomatic_70_79 = 0.15,
                    fraction_asymptomatic_80    = 0.15,
                    mild_fraction_0_9           = 0.00,
                    mild_fraction_10_19         = 0.00,
                    mild_fraction_20_29         = 0.00,
                    mild_fraction_30_39         = 0.00,
                    mild_fraction_40_49         = 0.00,
                    mild_fraction_50_59         = 0.00,
                    mild_fraction_60_69         = 0.00,
                    mild_fraction_70_79         = 0.00,
                    mild_fraction_80            = 0.00,
                    hospitalised_fraction_0_9  =0.50,
                    hospitalised_fraction_10_19=0.40,
                    hospitalised_fraction_20_29=0.30,
                    hospitalised_fraction_30_39=0.20,
                    hospitalised_fraction_40_49=0.20,
                    hospitalised_fraction_50_59=0.20,
                    hospitalised_fraction_60_69=0.30,
                    hospitalised_fraction_70_79=0.30,
                    hospitalised_fraction_80   =0.20,
                    critical_fraction_0_9 =0.50,
                    critical_fraction_10_19=0.40,
                    critical_fraction_20_29=0.30,
                    critical_fraction_30_39=0.20,
                    critical_fraction_40_49=0.20,
                    critical_fraction_50_59=0.20,
                    critical_fraction_60_69=0.30,
                    critical_fraction_70_79=0.30,
                    critical_fraction_80   =0.20,
                    fatality_fraction_0_9  =0.50,
                    fatality_fraction_10_19=0.40,
                    fatality_fraction_20_29=0.30,
                    fatality_fraction_30_39=0.20,
                    fatality_fraction_40_49=0.20,
                    fatality_fraction_50_59=0.20,
                    fatality_fraction_60_69=0.30,
                    fatality_fraction_70_79=0.30,
                    fatality_fraction_80   =0.20,
                )
            ),
            dict(
                test_params = dict(
                    fraction_asymptomatic_0_9   = 0.35,
                    fraction_asymptomatic_10_19 = 0.35,
                    fraction_asymptomatic_20_29 = 0.35,
                    fraction_asymptomatic_30_39 = 0.35,
                    fraction_asymptomatic_40_49 = 0.35,
                    fraction_asymptomatic_50_59 = 0.15,
                    fraction_asymptomatic_60_69 = 0.15,
                    fraction_asymptomatic_70_79 = 0.15,
                    fraction_asymptomatic_80    = 0.15,
                    mild_fraction_0_9           = 0.00,
                    mild_fraction_10_19         = 0.00,
                    mild_fraction_20_29         = 0.00,
                    mild_fraction_30_39         = 0.00,
                    mild_fraction_40_49         = 0.00,
                    mild_fraction_50_59         = 0.00,
                    mild_fraction_60_69         = 0.00,
                    mild_fraction_70_79         = 0.00,
                    mild_fraction_80            = 0.00,
                    hospitalised_fraction_0_9  =0.05,
                    hospitalised_fraction_10_19=0.05,
                    hospitalised_fraction_20_29=0.05,
                    hospitalised_fraction_30_39=0.20,
                    hospitalised_fraction_40_49=0.20,
                    hospitalised_fraction_50_59=0.20,
                    hospitalised_fraction_60_69=0.30,
                    hospitalised_fraction_70_79=0.80,
                    hospitalised_fraction_80   =0.90,
                    critical_fraction_0_9  =0.05,
                    critical_fraction_10_19=0.05,
                    critical_fraction_20_29=0.05,
                    critical_fraction_30_39=0.20,
                    critical_fraction_40_49=0.20,
                    critical_fraction_50_59=0.20,
                    critical_fraction_60_69=0.30,
                    critical_fraction_70_79=0.80,
                    critical_fraction_80   =0.90,
                    fatality_fraction_0_9  =0.05,
                    fatality_fraction_10_19=0.05,
                    fatality_fraction_20_29=0.05,
                    fatality_fraction_30_39=0.20,
                    fatality_fraction_40_49=0.20,
                    fatality_fraction_50_59=0.20,
                    fatality_fraction_60_69=0.30,
                    fatality_fraction_70_79=0.80,
                    fatality_fraction_80   =0.90,
                )
            ),
            dict(
                test_params = dict(
                    fraction_asymptomatic_0_9   = 0.70,
                    fraction_asymptomatic_10_19 = 0.70,
                    fraction_asymptomatic_20_29 = 0.60,
                    fraction_asymptomatic_30_39 = 0.50,
                    fraction_asymptomatic_40_49 = 0.40,
                    fraction_asymptomatic_50_59 = 0.30,
                    fraction_asymptomatic_60_69 = 0.20,
                    fraction_asymptomatic_70_79 = 0.10,
                    fraction_asymptomatic_80    = 0.10,
                    mild_fraction_0_9           = 0.00,
                    mild_fraction_10_19         = 0.00,
                    mild_fraction_20_29         = 0.00,
                    mild_fraction_30_39         = 0.00,
                    mild_fraction_40_49         = 0.00,
                    mild_fraction_50_59         = 0.00,
                    mild_fraction_60_69         = 0.00,
                    mild_fraction_70_79         = 0.00,
                    mild_fraction_80            = 0.00,
                    hospitalised_fraction_0_9=0.02,
                    hospitalised_fraction_10_19=0.02,
                    hospitalised_fraction_20_29=0.02,
                    hospitalised_fraction_30_39=0.10,
                    hospitalised_fraction_40_49=0.15,
                    hospitalised_fraction_50_59=0.20,
                    hospitalised_fraction_60_69=0.25,
                    hospitalised_fraction_70_79=0.30,
                    hospitalised_fraction_80  =0.50,
                    critical_fraction_0_9  =0.02,
                    critical_fraction_10_19=0.02,
                    critical_fraction_20_29=0.02,
                    critical_fraction_30_39=0.10,
                    critical_fraction_40_49=0.15,
                    critical_fraction_50_59=0.20,
                    critical_fraction_60_69=0.25,
                    critical_fraction_70_79=0.30,
                    critical_fraction_80  =0.50,
                    fatality_fraction_0_9 =0.02,
                    fatality_fraction_10_19=0.02,
                    fatality_fraction_20_29=0.02,
                    fatality_fraction_30_39=0.10,
                    fatality_fraction_40_49=0.15,
                    fatality_fraction_50_59=0.20,
                    fatality_fraction_60_69=0.25,
                    fatality_fraction_70_79=0.30,
                    fatality_fraction_80   =0.50,
                )
            ),
            dict(
                test_params = dict(
                    fraction_asymptomatic_0_9   = 0.70,
                    fraction_asymptomatic_10_19 = 0.70,
                    fraction_asymptomatic_20_29 = 0.40,
                    fraction_asymptomatic_30_39 = 0.40,
                    fraction_asymptomatic_40_49 = 0.40,
                    fraction_asymptomatic_50_59 = 0.40,
                    fraction_asymptomatic_60_69 = 0.30,
                    fraction_asymptomatic_70_79 = 0.20,
                    fraction_asymptomatic_80    = 0.20,
                    mild_fraction_0_9           = 0.00,
                    mild_fraction_10_19         = 0.00,
                    mild_fraction_20_29         = 0.00,
                    mild_fraction_30_39         = 0.00,
                    mild_fraction_40_49         = 0.00,
                    mild_fraction_50_59         = 0.00,
                    mild_fraction_60_69         = 0.00,
                    mild_fraction_70_79         = 0.00,
                    mild_fraction_80            = 0.00,
                    hospitalised_fraction_0_9  =0.20,
                    hospitalised_fraction_10_19=0.20,
                    hospitalised_fraction_20_29=0.20,
                    hospitalised_fraction_30_39=0.20,
                    hospitalised_fraction_40_49=0.20,
                    hospitalised_fraction_50_59=0.20,
                    hospitalised_fraction_60_69=0.20,
                    hospitalised_fraction_70_79=0.20,
                    hospitalised_fraction_80   =0.20,
                    critical_fraction_0_9  =0.20,
                    critical_fraction_10_19=0.20,
                    critical_fraction_20_29=0.20,
                    critical_fraction_30_39=0.20,
                    critical_fraction_40_49=0.20,
                    critical_fraction_50_59=0.20,
                    critical_fraction_60_69=0.20,
                    critical_fraction_70_79=0.20,
                    critical_fraction_80   =0.20,
                    fatality_fraction_0_9  =0.20,
                    fatality_fraction_10_19=0.20,
                    fatality_fraction_20_29=0.20,
                    fatality_fraction_30_39=0.20,
                    fatality_fraction_40_49=0.20,
                    fatality_fraction_50_59=0.20,
                    fatality_fraction_60_69=0.20,
                    fatality_fraction_70_79=0.20,
                    fatality_fraction_80   =0.20,
                )
            ),
            dict(
                test_params = dict(
                    fraction_asymptomatic_0_9   = 0.18,
                    fraction_asymptomatic_10_19 = 0.18,
                    fraction_asymptomatic_20_29 = 0.18,
                    fraction_asymptomatic_30_39 = 0.18,
                    fraction_asymptomatic_40_49 = 0.18,
                    fraction_asymptomatic_50_59 = 0.18,
                    fraction_asymptomatic_60_69 = 0.18,
                    fraction_asymptomatic_70_79 = 0.18,
                    fraction_asymptomatic_80    = 0.18,
                    mild_fraction_0_9           = 0.79,
                    mild_fraction_10_19         = 0.79,
                    mild_fraction_20_29         = 0.73,
                    mild_fraction_30_39         = 0.68,
                    mild_fraction_40_49         = 0.65,
                    mild_fraction_50_59         = 0.59,
                    mild_fraction_60_69         = 0.53,
                    mild_fraction_70_79         = 0.41,
                    mild_fraction_80            = 0.27,
                    hospitalised_fraction_0_9  =0.03,
                    hospitalised_fraction_10_19=0.08,
                    hospitalised_fraction_20_29=0.11,
                    hospitalised_fraction_30_39=0.19,
                    hospitalised_fraction_40_49=0.24,
                    hospitalised_fraction_50_59=0.36,
                    hospitalised_fraction_60_69=0.46,
                    hospitalised_fraction_70_79=0.48,
                    hospitalised_fraction_80   =0.41,
                    critical_fraction_0_9  =0.05,
                    critical_fraction_10_19=0.05,
                    critical_fraction_20_29=0.05,
                    critical_fraction_30_39=0.05,
                    critical_fraction_40_49=0.063,
                    critical_fraction_50_59=0.122,
                    critical_fraction_60_69=0.274,
                    critical_fraction_70_79=0.432,
                    critical_fraction_80   =0.709,
                    fatality_fraction_0_9  =0.33,
                    fatality_fraction_10_19=0.25,
                    fatality_fraction_20_29=0.5,
                    fatality_fraction_30_39=0.5,
                    fatality_fraction_40_49=0.5,
                    fatality_fraction_50_59=0.69,
                    fatality_fraction_60_69=0.65,
                    fatality_fraction_70_79=0.88,
                    fatality_fraction_80   =1,
                )
            ),
            dict(
                test_params = dict(
                    fraction_asymptomatic_0_9   = 0.10,
                    fraction_asymptomatic_10_19 = 0.10,
                    fraction_asymptomatic_20_29 = 0.10,
                    fraction_asymptomatic_30_39 = 0.10,
                    fraction_asymptomatic_40_49 = 0.10,
                    fraction_asymptomatic_50_59 = 0.10,
                    fraction_asymptomatic_60_69 = 0.10,
                    fraction_asymptomatic_70_79 = 0.10,
                    fraction_asymptomatic_80    = 0.10,
                    mild_fraction_0_9           = 0.10,
                    mild_fraction_10_19         = 0.10,
                    mild_fraction_20_29         = 0.10,
                    mild_fraction_30_39         = 0.10,
                    mild_fraction_40_49         = 0.10,
                    mild_fraction_50_59         = 0.10,
                    mild_fraction_60_69         = 0.10,
                    mild_fraction_70_79         = 0.10,
                    mild_fraction_80            = 0.10,
                    hospitalised_fraction_0_9  =0.20,
                    hospitalised_fraction_10_19=0.20,
                    hospitalised_fraction_20_29=0.20,
                    hospitalised_fraction_30_39=0.20,
                    hospitalised_fraction_40_49=0.20,
                    hospitalised_fraction_50_59=0.20,
                    hospitalised_fraction_60_69=0.20,
                    hospitalised_fraction_70_79=0.20,
                    hospitalised_fraction_80   =0.20,
                    critical_fraction_0_9  =0.20,
                    critical_fraction_10_19=0.20,
                    critical_fraction_20_29=0.20,
                    critical_fraction_30_39=0.20,
                    critical_fraction_40_49=0.20,
                    critical_fraction_50_59=0.20,
                    critical_fraction_60_69=0.20,
                    critical_fraction_70_79=0.20,
                    critical_fraction_80   =0.20,
                    fatality_fraction_0_9  =0.20,
                    fatality_fraction_10_19=0.20,
                    fatality_fraction_20_29=0.20,
                    fatality_fraction_30_39=0.20,
                    fatality_fraction_40_49=0.20,
                    fatality_fraction_50_59=0.20,
                    fatality_fraction_60_69=0.20,
                    fatality_fraction_70_79=0.20,
                    fatality_fraction_80   =0.20,
                )
            ),
        ],
        "test_recovered_susceptible_transition_time" : [dict()],
        "test_get_individuals": [dict(
            test_params=dict(
                n_total = 10000,
                end_time = 200
            )
        )],
        "test_multi_strain_disease_dynamics" : [
            dict(
                test_params = dict(
                    n_total  = 3e4,
                    end_time = 50,
                    n_seed_infection = 0,
                    infectious_rate  = 7,
                    max_n_strains = 2,
                    fraction_asymptomatic_0_9   = 0.00,
                    fraction_asymptomatic_10_19 = 0.00,
                    fraction_asymptomatic_20_29 = 0.00,
                    fraction_asymptomatic_30_39 = 0.00,
                    fraction_asymptomatic_40_49 = 0.00,
                    fraction_asymptomatic_50_59 = 0.00,
                    fraction_asymptomatic_60_69 = 0.00,
                    fraction_asymptomatic_70_79 = 0.00,
                    fraction_asymptomatic_80    = 0.00,
                    mild_fraction_0_9           = 0.00,
                    mild_fraction_10_19         = 0.00,
                    mild_fraction_20_29         = 0.00,
                    mild_fraction_30_39         = 0.00,
                    mild_fraction_40_49         = 0.00,
                    mild_fraction_50_59         = 0.00,
                    mild_fraction_60_69         = 0.00,
                    mild_fraction_70_79         = 0.00,
                    mild_fraction_80            = 0.00,
                    hospitalised_fraction_0_9  =0.2,
                    hospitalised_fraction_10_19=0.2,
                    hospitalised_fraction_20_29=0.2,
                    hospitalised_fraction_30_39=0.2,
                    hospitalised_fraction_40_49=0.2,
                    hospitalised_fraction_50_59=0.2,
                    hospitalised_fraction_60_69=0.2,
                    hospitalised_fraction_70_79=0.2,
                    hospitalised_fraction_80   =0.2
                ),
                hospitalised_fraction_strain_1 = [ 0, 0, 0, 0, 0, 0.8, 0.8, 0.8, 0.8 ]
            ),
        ],
        "test_multi_strain_disease_transition_times": [
            dict(
                test_params = dict(
                    n_total=100000,
                    n_seed_infection=200,               
                    end_time=30,
                    rebuild_networks=0,
                    infectious_rate=6.0,
                    max_n_strains=2,
                    mean_time_to_symptoms=4.0,
                    sd_time_to_symptoms=2.0,
                    mean_time_to_hospital=1.0,
                    mean_time_to_critical=1.0,
                    mean_time_to_recover=20.0,
                    sd_time_to_recover=8.0,
                    mean_time_to_death=12.0,
                    sd_time_to_death=5.0,
                    mean_asymptomatic_to_recovery=15.0,
                    sd_asymptomatic_to_recovery=5.0,
                    mean_time_hospitalised_recovery=6,
                    sd_time_hospitalised_recovery=3,
                    mean_time_critical_survive=4,
                    sd_time_critical_survive=2,
                    hospitalised_fraction_0_9  =0.2,
                    hospitalised_fraction_10_19=0.2,
                    hospitalised_fraction_20_29=0.2,
                    hospitalised_fraction_30_39=0.2,
                    hospitalised_fraction_40_49=0.2,
                    hospitalised_fraction_50_59=0.2,
                    hospitalised_fraction_60_69=0.2,
                    hospitalised_fraction_70_79=0.2,
                    hospitalised_fraction_80   =0.2,
                    critical_fraction_0_9  =0.50,
                    critical_fraction_10_19=0.50,
                    critical_fraction_20_29=0.50,
                    critical_fraction_30_39=0.50,
                    critical_fraction_40_49=0.50,
                    critical_fraction_50_59=0.50,
                    critical_fraction_60_69=0.50,
                    critical_fraction_70_79=0.50,
                    critical_fraction_80   =0.50,
                    fatality_fraction_0_9  =0.50,
                    fatality_fraction_10_19=0.50,
                    fatality_fraction_20_29=0.50,
                    fatality_fraction_30_39=0.50,
                    fatality_fraction_40_49=0.50,
                    fatality_fraction_50_59=0.50,
                    fatality_fraction_60_69=0.50,
                    fatality_fraction_70_79=0.50,
                    fatality_fraction_80   =0.50,
                ),
                strain1_params = dict(    
                    mean_time_to_symptoms=6.0,
                    sd_time_to_symptoms=3.0,
                    mean_time_to_hospital=1.8,
                    mean_time_to_critical=3.0,
                    mean_time_to_recover=12.0,
                    sd_time_to_recover=6.0,
                    mean_time_to_death=24.0,
                    sd_time_to_death=15,
                    mean_asymptomatic_to_recovery=10.0,
                    sd_asymptomatic_to_recovery=3.0,
                    mean_time_hospitalised_recovery=10,
                    sd_time_hospitalised_recovery=6,
                    mean_time_critical_survive=8,
                    sd_time_critical_survive=3.
                )
            ),
        ],
        "test_multi_strain_disease_outcome_proportions": [
            dict(
                test_params = dict(
                    n_total          = 100000,
                    n_seed_infection = 200,               
                    end_time         = 80,
                    rebuild_networks = 0,
                    infectious_rate  = 6.0,
                    max_n_strains    = 2,
                    population_0_9   = 10000,
                    population_10_19 = 10000,
                    population_20_29 = 10000,
                    population_30_39 = 10000,
                    population_40_49 = 10000,
                    population_50_59 = 10000,
                    population_60_69 = 10000,
                    population_70_79 = 10000,
                    population_80    = 10000,
                    fraction_asymptomatic_0_9   = 0.10,
                    fraction_asymptomatic_10_19 = 0.10,
                    fraction_asymptomatic_20_29 = 0.10,
                    fraction_asymptomatic_30_39 = 0.15,
                    fraction_asymptomatic_40_49 = 0.15,
                    fraction_asymptomatic_50_59 = 0.15,
                    fraction_asymptomatic_60_69 = 0.20,
                    fraction_asymptomatic_70_79 = 0.20,
                    fraction_asymptomatic_80    = 0.20,
                    mild_fraction_0_9           = 0.20,
                    mild_fraction_10_19         = 0.20,
                    mild_fraction_20_29         = 0.20,
                    mild_fraction_30_39         = 0.15,
                    mild_fraction_40_49         = 0.15,
                    mild_fraction_50_59         = 0.15,
                    mild_fraction_60_69         = 0.10,
                    mild_fraction_70_79         = 0.10,
                    mild_fraction_80            = 0.10,
                    hospitalised_fraction_0_9  =0.40,
                    hospitalised_fraction_10_19=0.40,
                    hospitalised_fraction_20_29=0.40,
                    hospitalised_fraction_30_39=0.60,
                    hospitalised_fraction_40_49=0.60,
                    hospitalised_fraction_50_59=0.60,
                    hospitalised_fraction_60_69=0.80,
                    hospitalised_fraction_70_79=0.80,
                    hospitalised_fraction_80   =0.80,
                    critical_fraction_0_9  =0.60,
                    critical_fraction_10_19=0.60,
                    critical_fraction_20_29=0.60,
                    critical_fraction_30_39=0.70,
                    critical_fraction_40_49=0.70,
                    critical_fraction_50_59=0.70,
                    critical_fraction_60_69=0.80,
                    critical_fraction_70_79=0.80,
                    critical_fraction_80   =0.80,
                    fatality_fraction_0_9  =0.40,
                    fatality_fraction_10_19=0.40,
                    fatality_fraction_20_29=0.40,
                    fatality_fraction_30_39=0.30,
                    fatality_fraction_40_49=0.30,
                    fatality_fraction_50_59=0.30,
                    fatality_fraction_60_69=0.20,
                    fatality_fraction_70_79=0.20,
                    fatality_fraction_80   =0.20,
                ),
                strain1_params = dict(
                    fraction_asymptomatic = [ 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15 ],
                    mild_fraction         = [ 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.15, 0.15, 0.15 ],
                    hospitalised_fraction = [ 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4 ],
                    critical_fraction     = [ 0.8, 0.8, 0.8, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6 ],
                    fatality_fraction     = [ 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8 ]
                )
            ),
        ],
        "test_multi_strain_infectious_factor": [
            dict(
                test_params=dict(
                    n_total=100000,
                    n_seed_infection=200,
                    end_time=30,
                    rebuild_networks=0,
                    infectious_rate=18.0,
                    max_n_strains=3,
                    mild_infectious_factor = 0,
                    asymptomatic_infectious_factor = 0,
                    fraction_asymptomatic_0_9=0.33,
                    fraction_asymptomatic_10_19=0.33,
                    fraction_asymptomatic_20_29=0.33,
                    fraction_asymptomatic_30_39=0.33,
                    fraction_asymptomatic_40_49=0.33,
                    fraction_asymptomatic_50_59=0.33,
                    fraction_asymptomatic_60_69=0.33,
                    fraction_asymptomatic_70_79=0.33,
                    fraction_asymptomatic_80=0.33,
                    mild_fraction_0_9=0.33,
                    mild_fraction_10_19=0.33,
                    mild_fraction_20_29=0.33,
                    mild_fraction_30_39=0.33,
                    mild_fraction_40_49=0.33,
                    mild_fraction_50_59=0.33,
                    mild_fraction_60_69=0.33,
                    mild_fraction_70_79=0.33,
                    mild_fraction_80=0.33,
                )
            ),
        ],
    }
    """
    Test class for checking
    """
    def test_zero_recovery(self):
        """
        Setting recover times to be very large should avoid seeing any in cumulative recovered compartment ('n_recovered')
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
        for time in range(params.get_param("end_time")):
            model.one_time_step()
        
        np.testing.assert_equal(
            model.one_time_step_results()["n_recovered"],
            0)

    def test_zero_deaths(self):
        """
        Set fatality ratio to zero, should have no deaths if always places in the ICU
        """
        params = utils.get_params_swig()
        params.set_param("n_total", 10000)
        params = utils.set_fatality_fraction_all(params, 0.0)
        params = utils.set_location_death_icu_all(params, 1.0)
        
        # Call the model
        model  = utils.get_model_swig( params )
        for time in range(params.get_param("end_time")):
            model.one_time_step()

        np.testing.assert_equal(
            model.one_time_step_results()["n_death"],
            0)

    def test_total_infectious_rate_zero(self):
        """
        Set infectious rate to zero results in only "n_seed_infection" as total_infected
        """
        params = utils.get_params_swig()
        params.set_param("n_total", 10000)
        params.set_param("infectious_rate", 0.0)

        # Call the model
        model  = utils.get_model_swig( params )
        for time in range(params.get_param("end_time")):
            model.one_time_step()
        
        np.testing.assert_equal(
            model.one_time_step_results()["total_infected"], 
            int(params.get_param("n_seed_infection")))

    def test_zero_infected(self):
        """
        Set seed-cases to zero should result in zero sum of output column
        """
        params = utils.get_params_swig()
        params.set_param("n_seed_infection", 0)
        params.set_param("n_total", 10000)
        params.set_param("end_time", 100)

        # Call the model
        model  = utils.get_model_swig( params )
        
        for _ in range(params.get_param("end_time")):
            model.one_time_step()

        np.testing.assert_equal(model.one_time_step_results()["total_infected"], 0)

    def test_disease_transition_times( self, test_params ):
        """
        Test that the mean and standard deviation of the transition times between
        states agrees with the parameters
        """
        std_error_limit = 4

        params = utils.get_params_swig()
        params = utils.turn_off_interventions(params, 50)
        params.set_param("n_total", 50000)
        params.set_param("n_seed_infection", 200)
        params.set_param("end_time", 30)
        params.set_param("infectious_rate", 6.0)
        params.set_param("hospital_on", 0) #turning off hospital as this affects disease transitions
        for param, value in test_params.items():
            params.set_param( param, value )

        model  = utils.get_model_swig( params )
        for _ in range(params.get_param("end_time")):
            model.one_time_step()
        
        model.write_transmissions()
        model.write_individual_file()

        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_indiv = pd.merge(df_indiv, df_trans,
            left_on = "ID", right_on = "ID_recipient", how = "left")

        # time infected until showing symptoms
        df_indiv["t_p_s"] = df_indiv["time_symptomatic"] - df_indiv["time_infected"]
        mean = df_indiv[
            (df_indiv["time_infected"] > 0) & (df_indiv["time_asymptomatic"] < 0)
        ]["t_p_s"].mean()
        sd = df_indiv[
            (df_indiv["time_infected"] > 0) & (df_indiv["time_asymptomatic"] < 0)
        ]["t_p_s"].std()
        N = len(
            df_indiv[
                (df_indiv["time_infected"] > 0) & (df_indiv["time_asymptomatic"] < 0)
            ]
        )
        np.testing.assert_allclose(
            mean, test_params[ "mean_time_to_symptoms" ], atol=std_error_limit * sd / sqrt(N)
        )
        np.testing.assert_allclose(
            sd, test_params[ "sd_time_to_symptoms" ], atol=std_error_limit * sd / sqrt(N)
        )

        # time showing symptoms until going to hospital
        df_indiv["t_s_h"] = df_indiv["time_hospitalised"] - df_indiv["time_symptomatic"]
        mean = df_indiv[(df_indiv["time_hospitalised"] > 0)]["t_s_h"].mean()
        sd = df_indiv[(df_indiv["time_hospitalised"] > 0)]["t_s_h"].std()
        N = len(df_indiv[(df_indiv["time_hospitalised"] > 0)])
        np.testing.assert_allclose(
            mean, test_params[ "mean_time_to_hospital" ], atol=std_error_limit * sd / sqrt(N)
        )

        # time hospitalised until moving to the ICU
        df_indiv["t_h_c"] = df_indiv["time_critical"] - df_indiv["time_hospitalised"]
        mean = df_indiv[(df_indiv["time_critical"] > 0)]["t_h_c"].mean()
        sd = df_indiv[(df_indiv["time_critical"] > 0)]["t_h_c"].std()
        N = len(df_indiv[(df_indiv["time_critical"] > 0)])
        np.testing.assert_allclose(
            mean, test_params[ "mean_time_to_critical" ], atol=std_error_limit * sd / sqrt(N)
        )

        # time from symptoms to recover if not hospitalised
        df_indiv["t_s_r"] = df_indiv["time_recovered"] - df_indiv["time_symptomatic"]
        mean = df_indiv[
            (df_indiv["time_recovered"] > 0)
            & (df_indiv["time_asymptomatic"] < 0)
            & (df_indiv["time_hospitalised"] <  0)
        ]["t_s_r"].mean()
        sd = df_indiv[
            (df_indiv["time_recovered"] > 0)
            & (df_indiv["time_asymptomatic"] < 0)
            & (df_indiv["time_hospitalised"] < 0)
        ]["t_s_r"].std()
        N = len(
            df_indiv[
                (df_indiv["time_recovered"] > 0)
                & (df_indiv["time_asymptomatic"] < 0)
                & (df_indiv["time_hospitalised"] < 0)
            ]
        )
        np.testing.assert_allclose(
            mean, test_params[ "mean_time_to_recover" ], atol=std_error_limit * sd / sqrt(N)
        )
        np.testing.assert_allclose(
            sd, test_params[ "sd_time_to_recover" ], atol=std_error_limit * sd / sqrt(N)
        )

        # time from hospitalised to recover if don't got to ICU
        df_indiv["t_h_r"] = df_indiv["time_recovered"] - df_indiv["time_hospitalised"]
        mean = df_indiv[
            (df_indiv["time_recovered"] > 0)
            & (df_indiv["time_critical"] < 0)
            & (df_indiv["time_hospitalised"] > 0)
        ]["t_h_r"].mean()
        sd = df_indiv[
            (df_indiv["time_recovered"] > 0)
            & (df_indiv["time_critical"] < 0)
            & (df_indiv["time_hospitalised"] > 0)
        ]["t_h_r"].std()
        N = len(
            df_indiv[
                (df_indiv["time_recovered"] > 0)
                & (df_indiv["time_critical"] < 0)
                & (df_indiv["time_hospitalised"] > 0)
            ]
        )
        np.testing.assert_allclose(
            mean, test_params[ "mean_time_hospitalised_recovery" ], atol=std_error_limit * sd / sqrt(N)
        )
        np.testing.assert_allclose(
            sd, test_params[ "sd_time_hospitalised_recovery" ], atol=std_error_limit * sd / sqrt(N)
        )

        # time in ICU
        df_indiv["t_c_r"] = df_indiv["time_hospitalised_recovering"] - df_indiv["time_critical"]
        mean = df_indiv[
            (df_indiv["time_hospitalised_recovering"] > 0) & (df_indiv["time_critical"] > 0)
        ]["t_c_r"].mean()
        sd = df_indiv[
            (df_indiv["time_hospitalised_recovering"] > 0) & (df_indiv["time_critical"] > 0)
        ]["t_c_r"].std()
        N = len(
            df_indiv[(df_indiv["time_hospitalised_recovering"] > 0) & (df_indiv["time_critical"] > 0)]
        )
        np.testing.assert_allclose(
            mean, test_params[ "mean_time_critical_survive" ], atol=std_error_limit * sd / sqrt(N)
        )
        np.testing.assert_allclose(
            sd, test_params[ "sd_time_critical_survive" ], atol=std_error_limit * sd / sqrt(N)
        )

        # time from ICU to death
        df_indiv["t_c_d"] = df_indiv["time_death"] - df_indiv["time_critical"]
        mean = df_indiv[
            (df_indiv["time_death"] > 0) & (df_indiv["time_critical"] > 0)
        ]["t_c_d"].mean()
        sd = df_indiv[
            (df_indiv["time_death"] > 0) & (df_indiv["time_critical"] > 0)
        ]["t_c_d"].std()
        N = len( df_indiv[
            (df_indiv["time_death"] > 0) & (df_indiv["time_critical"] > 0)
        ] )
        np.testing.assert_allclose(
            mean, test_params[ "mean_time_to_death" ], atol=std_error_limit * sd / sqrt(N)
        )
        np.testing.assert_allclose(
            sd, test_params[ "sd_time_to_death" ], atol=std_error_limit * sd / sqrt(N)
        )

        # time from asymptomatic to recover
        df_indiv["t_a_r"] = df_indiv["time_recovered"] - df_indiv["time_asymptomatic"]
        mean = df_indiv[
            (df_indiv["time_recovered"] > 0) & (df_indiv["time_asymptomatic"] > 0)
        ]["t_a_r"].mean()
        sd = df_indiv[
            (df_indiv["time_recovered"] > 0) & (df_indiv["time_asymptomatic"] > 0)
        ]["t_a_r"].std()
        N = len(
            df_indiv[
                (df_indiv["time_recovered"] > 0) & (df_indiv["time_asymptomatic"] > 0)
            ]
        )
        np.testing.assert_allclose(
            mean, test_params[ "mean_asymptomatic_to_recovery" ], atol=std_error_limit * sd / sqrt(N)
        )
        np.testing.assert_allclose(
            sd, test_params[ "sd_asymptomatic_to_recovery" ], atol=std_error_limit * sd / sqrt(N)
        )

    def test_disease_outcome_proportions( self, test_params ):
        """
        Test that the fraction of infected people following each path for
        the progression of the disease agrees with the parameters
        """
        std_error_limit = 5

        params = utils.get_params_swig()
        params = utils.turn_off_interventions(params, 50)

        params.set_param("n_total", 20000)
        params.set_param("n_seed_infection", 200)
        params.set_param("end_time", 250)
        params.set_param("infectious_rate", 4.0)
        params.set_param("mild_infectious_factor", 1.0)
        params.set_param("hospital_on", 0)
        for param, value in test_params.items():
            params.set_param( param, value )

        fraction_asymptomatic = [
            test_params[ "fraction_asymptomatic_0_9" ],
            test_params[ "fraction_asymptomatic_10_19" ],
            test_params[ "fraction_asymptomatic_20_29" ],
            test_params[ "fraction_asymptomatic_30_39" ],
            test_params[ "fraction_asymptomatic_40_49" ],
            test_params[ "fraction_asymptomatic_50_59" ],
            test_params[ "fraction_asymptomatic_60_69" ],
            test_params[ "fraction_asymptomatic_70_79" ],
            test_params[ "fraction_asymptomatic_80" ],
        ]

        mild_fraction = [
            test_params[ "mild_fraction_0_9" ],
            test_params[ "mild_fraction_10_19" ],
            test_params[ "mild_fraction_20_29" ],
            test_params[ "mild_fraction_30_39" ],
            test_params[ "mild_fraction_40_49" ],
            test_params[ "mild_fraction_50_59" ],
            test_params[ "mild_fraction_60_69" ],
            test_params[ "mild_fraction_70_79" ],
            test_params[ "mild_fraction_80" ],
        ]

        hospitalised_fraction = [
            test_params[ "hospitalised_fraction_0_9" ],
            test_params[ "hospitalised_fraction_10_19" ],
            test_params[ "hospitalised_fraction_20_29" ],
            test_params[ "hospitalised_fraction_30_39" ],
            test_params[ "hospitalised_fraction_40_49" ],
            test_params[ "hospitalised_fraction_50_59" ],
            test_params[ "hospitalised_fraction_60_69" ],
            test_params[ "hospitalised_fraction_70_79" ],
            test_params[ "hospitalised_fraction_80" ],
        ]

        critical_fraction = [
            test_params[ "critical_fraction_0_9" ],
            test_params[ "critical_fraction_10_19" ],
            test_params[ "critical_fraction_20_29" ],
            test_params[ "critical_fraction_30_39" ],
            test_params[ "critical_fraction_40_49" ],
            test_params[ "critical_fraction_50_59" ],
            test_params[ "critical_fraction_60_69" ],
            test_params[ "critical_fraction_70_79" ],
            test_params[ "critical_fraction_80" ],
        ]

        fatality_fraction = [
            test_params[ "fatality_fraction_0_9" ],
            test_params[ "fatality_fraction_10_19" ],
            test_params[ "fatality_fraction_20_29" ],
            test_params[ "fatality_fraction_30_39" ],
            test_params[ "fatality_fraction_40_49" ],
            test_params[ "fatality_fraction_50_59" ],
            test_params[ "fatality_fraction_60_69" ],
            test_params[ "fatality_fraction_70_79" ],
            test_params[ "fatality_fraction_80" ],
        ]

        model  = utils.get_model_swig( params )
        for _ in range(params.get_param("end_time")):
            model.one_time_step()
        
        model.write_transmissions()
        model.write_individual_file()

        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_indiv = pd.merge(df_indiv, df_trans,
            left_on = "ID", right_on = "ID_recipient", how = "left")

        # fraction asymptomatic vs mild+symptomatc
        N_asym = len(
            df_indiv[
                (df_indiv["time_infected"] > 0) & (df_indiv["time_asymptomatic"] > 0)
            ]
        )
        N_sym = len(
            df_indiv[
                (df_indiv["time_infected"] > 0) & (df_indiv["time_presymptomatic_severe"] > 0)
            ]
        )
        N_mild = len(
            df_indiv[
                (df_indiv["time_infected"] > 0) & (df_indiv["time_presymptomatic_mild"] > 0)
            ]
        )
        N = N_sym + N_asym + N_mild
        mean = N_asym / N
        sd = sqrt(mean * (1 - mean))
     #   np.testing.assert_allclose(
     #       mean, fraction_asymptomatic, atol=std_error_limit * sd / sqrt(N)
      #  )

        # asymptomatic fraction by age
        N_asymp_tot = 0
        N_symp_tot = 0
        N_mild_tot = 0
        asypmtomatic_fraction_weighted = 0
        mild_fraction_weighted = 0
        for idx in range( constant.N_AGE_GROUPS ):

            N_asymp = len(
                df_indiv[
                    (df_indiv["time_infected"] > 0) & (df_indiv["time_asymptomatic"] > 0)
                    & (df_indiv["age_group"] == constant.AGES[idx])
                ]
            )
            N_symp = len(
                df_indiv[
                    (df_indiv["time_presymptomatic"] > 0) & (df_indiv["time_presymptomatic_severe"] > 0)
                    & (df_indiv["age_group"] == constant.AGES[idx])
                ]
            )
            N_mild = len(
                df_indiv[
                    (df_indiv["time_presymptomatic"] > 0) & (df_indiv["time_presymptomatic_mild"] > 0)
                    & (df_indiv["age_group"] == constant.AGES[idx])
                ]
            )

            N    = N_symp + N_asymp + N_mild
            mean_asym = N_asymp / N
            sd_asym   = sqrt(mean_asym * (1 - mean_asym))
            np.testing.assert_allclose(
                mean_asym,
                fraction_asymptomatic[idx],
                atol=std_error_limit * sd_asym / sqrt( N ),
            )

            mean_mild = N_mild / N
            sd_mild   = sqrt(mean_mild * (1 - mean_mild))
            np.testing.assert_allclose(
                mean_mild,
                mild_fraction[idx],
                atol=std_error_limit * sd_mild / sqrt( N ),
            )

            N_asymp_tot += N_asymp
            N_symp_tot  += N_symp
            N_mild_tot += N_mild
            asypmtomatic_fraction_weighted += fraction_asymptomatic[idx] * N
            mild_fraction_weighted += mild_fraction[idx] * N

        # overall asymptomatic and mild fractions
        N    = N_symp_tot + N_asymp_tot + N_mild_tot
        mean_asym = N_asymp_tot / N
        sd_asym   = sqrt(mean_asym * (1 - mean_asym))
        asypmtomatic_fraction_weighted = asypmtomatic_fraction_weighted / N
        np.testing.assert_allclose(
            mean_asym,
            asypmtomatic_fraction_weighted,
            atol=std_error_limit * sd_asym / sqrt(N),
        )

        mean_mild = N_mild_tot / N
        sd_mild   = sqrt(mean_mild * (1 - mean_mild))
        mild_fraction_weighted = mild_fraction_weighted / N
        np.testing.assert_allclose(
            mean_mild,
            mild_fraction_weighted,
            atol=std_error_limit * sd_mild / sqrt(N),
        )

        # hospitalised fraction by age
        N_hosp_tot = 0
        N_symp_tot = 0
        hospitalised_fraction_weighted = 0
        for idx in range(constant.N_AGE_GROUPS):

            N_hosp = len(
                df_indiv[
                    (df_indiv["time_hospitalised"] > 0)
                    & (df_indiv["age_group"] == constant.AGES[idx])
                ]
            )
            N_symp = len(
                df_indiv[
                    (df_indiv["time_symptomatic_severe"] > 0)
                    & (df_indiv["age_group"] == constant.AGES[idx])
                ]
            )
            mean = N_hosp / N_symp
            sd = sqrt( max( mean * (1 - mean), hospitalised_fraction[idx] * ( 1 - hospitalised_fraction[idx] ) ) )
            np.testing.assert_allclose(
                mean,
                hospitalised_fraction[idx],
                atol=std_error_limit * sd / sqrt(N_symp),
            )

            N_hosp_tot += N_hosp
            N_symp_tot += N_symp
            hospitalised_fraction_weighted += hospitalised_fraction[idx] * N_symp

        # overall hospitalised fraction
        mean = N_hosp_tot / N_symp_tot
        sd = sqrt(mean * (1 - mean))
        hospitalised_fraction_weighted = hospitalised_fraction_weighted / N_symp_tot
        np.testing.assert_allclose(
            mean,
            hospitalised_fraction_weighted,
            atol=std_error_limit * sd / sqrt(N_symp_tot),
        )

        # critical fraction by age
        N_crit_tot = 0
        N_hosp_tot = 0
        critical_fraction_weighted = 0
        for idx in range(constant.N_AGE_GROUPS):

            N_crit = len(
                df_indiv[
                    ( (df_indiv["time_critical"] > 0) | (df_indiv["time_death"] > 0) )
                    & (df_indiv["age_group"] == constant.AGES[idx])
                ]
            )
            N_hosp = len(
                df_indiv[
                    (df_indiv["time_hospitalised"] > 0)
                    & (df_indiv["age_group"] == constant.AGES[idx])
                ]
            )

            if N_hosp > 0:
                mean = N_crit / N_hosp
                sd = sqrt(critical_fraction[idx]* (1 - critical_fraction[idx]))
                if N_crit > 0:
                    np.testing.assert_allclose(
                        mean,
                        critical_fraction[idx],
                        atol=std_error_limit * sd / sqrt(N_hosp),
                    )

            N_crit_tot += N_crit
            N_hosp_tot += N_hosp
            critical_fraction_weighted += critical_fraction[idx] * N_hosp

        # overall critical fraction
        mean = N_crit_tot / N_hosp_tot
        sd = sqrt(mean * (1 - mean))
        critical_fraction_weighted = critical_fraction_weighted / N_hosp_tot
        np.testing.assert_allclose(
            mean,
            critical_fraction_weighted,
            atol=std_error_limit * sd / sqrt(N_hosp_tot),
        )

        # critical fraction who die by age go to the ICU
        N_dead_tot = 0
        N_crit_tot = 0
        fatality_fraction_weighted = 0
        for idx in range(constant.N_AGE_GROUPS):

            N_dead = len(
                df_indiv[
                    (df_indiv["time_death"] > 0) &
                    (df_indiv["time_critical"] > 0) &
                    (df_indiv["age_group"] == constant.AGES[idx])
                ]
            )
            N_crit = len(
                df_indiv[
                    (df_indiv["time_critical"] > 0)
                    & (df_indiv["age_group"] == constant.AGES[idx])
                ]
            )

            if N_crit > 0:
                mean = N_dead / N_crit
                sd = sqrt(fatality_fraction[idx]* (1 - fatality_fraction[idx]))
                if N_dead > 0:
                    np.testing.assert_allclose(
                        mean,
                        fatality_fraction[idx],
                        atol=std_error_limit * sd / sqrt(N_crit),
                    )

            N_dead_tot += N_dead
            N_crit_tot += N_crit
            fatality_fraction_weighted += fatality_fraction[idx] * N_crit

        mean = N_dead_tot / N_crit_tot
        sd = sqrt(mean * (1 - mean))
        fatality_fraction_weighted = fatality_fraction_weighted / N_crit_tot
        np.testing.assert_allclose(
            mean,
            fatality_fraction_weighted,
            atol=std_error_limit * sd / sqrt(N_crit_tot),
        )
    
    def test_recovered_susceptible_transition_time(self):
        """
        Test that the recovered-susceptible transition times are as expected.  
        """
        
        params = utils.get_params_swig()
        params.set_param("n_total", 40000)
        params.set_param("mean_time_to_susceptible_after_shift", 5)
        params.set_param("time_to_susceptible_shift", 1)
        params.set_param("end_time", 200)
        
        model  = utils.get_model_swig( params )
        
        for time in range(params.get_param("end_time")):
            model.one_time_step()
        
        model.write_transmissions()
        model.write_individual_file()
        
        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        
        # Subset to those 1) ever recovered, 2) have transitioned from recovered->susceptible
        # at the end of the simulation.  
        df_sub = df_trans.loc[(df_trans.time_recovered > 0) & \
            (df_trans.time_susceptible - df_trans.time_recovered >= 0)]
        
        obs_recovered_suscept_time_mean = np.mean(df_sub.time_susceptible - df_sub.time_recovered)
        
        obs_recovered_suscept_time = np.mean(df_sub.time_susceptible - df_sub.time_recovered)
        exp_recovered_suscept_time = \
            params.get_param("mean_time_to_susceptible_after_shift") + \
            params.get_param("time_to_susceptible_shift")
        
        obs_recovered_suscept_time_min = np.min(df_sub.time_susceptible - df_sub.time_recovered)
        
        # Minimum time to susecptible (after time_to_susceptible_shift) is 1 day
        np.testing.assert_equal(obs_recovered_suscept_time_min, \
            params.get_param("time_to_susceptible_shift") + 1)
        
        np.testing.assert_almost_equal(obs_recovered_suscept_time_mean, exp_recovered_suscept_time, 
            decimal = 1)

    def test_get_individuals( self, test_params ):
        """
        Test that a dataframe of individuals is concordance with the individual/trans files
        """
        
        n_total = test_params["n_total"]
        end_time = test_params["end_time"]
        
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )
        model  = utils.get_model_swig( params )
        
        df_indiv_list = list()
        
        # Simulate for long enough for there to be some COVID-19 related mortality
        for time in range(end_time):
            # Every 5 years, save the dataframe of individuals
            if time % 5 == 0:
                
                df_all = model.get_individuals()
                df_alive = df_all[df_all.current_status != constant.EVENT_TYPES.DEATH.value]
                df_indiv_list.append(df_alive)
            
            model.one_time_step()
        
        # Write and read individual and transmission files
        model.write_individual_file()
        model.write_transmissions()
        
        # Pull individual file and convert to numpy array
        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_indiv = pd.merge(df_indiv, df_trans,
            left_on = "ID", right_on = "ID_recipient", how = "left")
        
        df_indiv.time_death = df_indiv.time_death.fillna(-1)
        df_alive_indiv = df_indiv.loc[
            df_indiv.time_death == -1, 
            ["ID", "current_status", "age_group", "occupation_network", "house_no",
                "infection_count", "vaccine_status"]
        ]
        # Check some individuals have died
        np.testing.assert_equal(model.one_time_step_results()["n_death"] > 0, True)
        
        cols2compare = ["ID", "age_group", "occupation_network", "house_no"]
        
        # Every 5 years, check the number of individuals alive is consistent across
        # both approaches
        for time in range(end_time):
            if time % 5 == 0:
                # Find alive individuals using the get_alive() method, convert to np array
                df_alive_t = df_indiv_list[time//5][cols2compare]
                array_alive = df_alive_t.to_numpy()
                
                # Find alive individuals using the individual+transmission file, convert to np array
                df_alive_indiv = df_indiv.loc[
                    (df_indiv.time_death > time) | (df_indiv.time_death == -1), cols2compare]
                array_alive_indiv = df_alive_indiv.to_numpy()
                
                np.testing.assert_array_equal(array_alive, array_alive_indiv)
                
    def test_multi_strain_disease_dynamics(self, test_params, hospitalised_fraction_strain_1 ) :
        """
        Test that checks the hospitalised fraction for the correct strain is realised
        """
        # set the np seed so the results are reproducible
        np.random.seed(0)
        
        sd_tol = 3
  
        hospitalised_fraction_strain_0 = [ 
            test_params[ "hospitalised_fraction_0_9" ],
            test_params[ "hospitalised_fraction_10_19" ],
            test_params[ "hospitalised_fraction_20_29" ],
            test_params[ "hospitalised_fraction_30_39" ],
            test_params[ "hospitalised_fraction_40_49" ],
            test_params[ "hospitalised_fraction_50_59" ],
            test_params[ "hospitalised_fraction_60_69" ],
            test_params[ "hospitalised_fraction_70_79" ],
            test_params[ "hospitalised_fraction_80" ],
        ]
        
        params = utils.get_params_swig()
        for param, value in test_params.items():
            params.set_param( param, value )
        model  = utils.get_model_swig( params )

        new_strain = model.add_new_strain( 1, hospitalised_fraction = hospitalised_fraction_strain_1 )

        n_seed_infection = 10
        n_total  = params.get_param( "n_total" )
        seeds    = np.random.choice( n_total, n_seed_infection * 2, replace=False)
        for idx in range( n_seed_infection ) :
            model.seed_infect_by_idx( seeds[ idx ] )
            model.seed_infect_by_idx( seeds[ idx + n_seed_infection], strain = new_strain )
            
        model.run( verbose = False )
        model.write_transmissions()
        df_trans = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_trans[ "hospitalised" ] = ( df_trans[ "time_hospitalised" ] > 0 )
              
        for age in range( len( AgeGroupEnum) ) :
            n_inf_0  = sum( ( df_trans[ "strain_idx"] == 0 ) & (df_trans[ "age_group_recipient" ] == age ) )
            n_hosp_0 = sum( ( df_trans[ "strain_idx"] == 0 ) & (df_trans[ "age_group_recipient" ] == age ) & df_trans[ "hospitalised" ] )
        
            mu_hosp_0 = n_hosp_0 / n_inf_0
            sd_hosp_0 = sqrt( hospitalised_fraction_strain_0[ age ] * ( 1 -  hospitalised_fraction_strain_0[ age ] ) / n_hosp_0 )
                    
            np.testing.assert_(n_inf_0 > 100, "insufficient infections to test properly")
            np.testing.assert_allclose(mu_hosp_0, hospitalised_fraction_strain_0[ age ], atol = sd_tol * sd_hosp_0, 
                                       err_msg = "incorrect fraction of hospitalisation for base strain in age group")        

            n_inf_1  = sum( ( df_trans[ "strain_idx"] == 1 ) & (df_trans[ "age_group_recipient" ] == age ) )
            n_hosp_1 = sum( ( df_trans[ "strain_idx"] == 1 ) & (df_trans[ "age_group_recipient" ] == age ) & df_trans[ "hospitalised" ] )
        
            mu_hosp_1 = n_hosp_1 / n_inf_1
            sd_hosp_1 = sqrt( hospitalised_fraction_strain_1[ age ] * ( 1 -  hospitalised_fraction_strain_1[ age ] ) / n_hosp_0 )
                    
            np.testing.assert_(n_inf_0 > 100, "insufficient infections to test properly")
            np.testing.assert_allclose(mu_hosp_1, hospitalised_fraction_strain_1[ age ], atol = sd_tol * sd_hosp_1, 
                                       err_msg = "incorrect fraction of hospitalisation for base strain in age group")

    def test_multi_strain_disease_transition_times( self, test_params, strain1_params ):
        """
        Test that the mean and standard deviation of the transition times between
        states agrees with the parameters for 2 strains
        """
        # set the np seed so the results are reproducible
        np.random.seed(0)
        std_error_limit = 4
        

        # add a new strain and seed the infection
        model      = abm.Model( params = test_params )
        new_strain = model.add_new_strain( 1, 
            mean_time_to_symptoms = strain1_params[ "mean_time_to_symptoms" ],
            sd_time_to_symptoms   = strain1_params[ "sd_time_to_symptoms" ],
            mean_time_to_hospital = strain1_params[ "mean_time_to_hospital" ],
            mean_time_to_critical = strain1_params[ "mean_time_to_critical" ],
            mean_time_to_recover  = strain1_params[ "mean_time_to_recover" ],
            sd_time_to_recover    = strain1_params[ "sd_time_to_recover" ],
            mean_time_to_death    = strain1_params[ "mean_time_to_death" ],
            sd_time_to_death      = strain1_params[ "sd_time_to_death" ],  
            mean_asymptomatic_to_recovery   = strain1_params[ "mean_asymptomatic_to_recovery" ],
            sd_asymptomatic_to_recovery     = strain1_params[ "sd_asymptomatic_to_recovery" ],
            mean_time_hospitalised_recovery = strain1_params[ "mean_time_hospitalised_recovery" ],
            sd_time_hospitalised_recovery   = strain1_params[ "sd_time_hospitalised_recovery" ],
            mean_time_critical_survive      = strain1_params[ "mean_time_critical_survive" ],
            sd_time_critical_survive        = strain1_params[ "sd_time_critical_survive" ],
        )
        seeds      = np.random.choice( test_params["n_total"], test_params["n_seed_infection"], replace=False)
        for idx in range( test_params["n_seed_infection"] ) :
            model.seed_infect_by_idx( seeds[ idx ], strain = new_strain )
                 
        # run the model and look at the infections
        model.run( verbose = False )
        df_trans = model.get_transmissions()

        # time infected until showing symptoms
        df_trans["t_p_s"] = df_trans["time_symptomatic"] - df_trans["time_infected"]
        df_t_p_s = df_trans[(df_trans["time_infected"] > 0) & (df_trans["time_asymptomatic"] < 0) ]
        t_p_s    = df_t_p_s[ df_t_p_s["strain_idx"] == 0 ]["t_p_s"]
        t_p_s_1  = df_t_p_s[ df_t_p_s["strain_idx"] == 1 ]["t_p_s"]
        np.testing.assert_allclose(t_p_s.mean(),test_params[ "mean_time_to_symptoms" ], atol=std_error_limit * t_p_s.std() / sqrt(len(t_p_s)))
        np.testing.assert_allclose(t_p_s.std(), test_params[ "sd_time_to_symptoms" ],   atol=std_error_limit * t_p_s.std() / sqrt(len(t_p_s)))
        np.testing.assert_allclose(t_p_s_1.mean(),strain1_params[ "mean_time_to_symptoms" ], atol=std_error_limit * t_p_s_1.std() / sqrt(len(t_p_s_1)))
        np.testing.assert_allclose(t_p_s_1.std(), strain1_params[ "sd_time_to_symptoms" ],   atol=std_error_limit * t_p_s_1.std() / sqrt(len(t_p_s_1)))

        # time showing symptoms until going to hospital
        df_trans["t_s_h"] = df_trans["time_hospitalised"] - df_trans["time_symptomatic"]
        df_t_s_h = df_trans[(df_trans["time_hospitalised"] > 0)]
        t_s_h    = df_t_s_h[ df_t_s_h["strain_idx"]==0]["t_s_h"] 
        t_s_h_1  = df_t_s_h[ df_t_s_h["strain_idx"]==1]["t_s_h"] 
        np.testing.assert_allclose(t_s_h.mean(),   test_params[ "mean_time_to_hospital" ],    atol=std_error_limit * t_s_h.std() / sqrt(len(t_s_h)))
        np.testing.assert_allclose(t_s_h_1.mean(), strain1_params[ "mean_time_to_hospital" ], atol=std_error_limit * t_s_h_1.std() / sqrt(len(t_s_h_1)))

        # time hospitalised until moving to the ICU
        df_trans["t_h_c"] = df_trans["time_critical"] - df_trans["time_hospitalised"]
        df_t_h_c = df_trans[(df_trans["time_critical"] > 0)];
        t_h_c   = df_t_h_c[df_t_h_c["strain_idx"]==0]["t_h_c"];
        t_h_c_1 = df_t_h_c[df_t_h_c["strain_idx"]==1]["t_h_c"];
        np.testing.assert_allclose(t_h_c.mean(),   test_params[ "mean_time_to_critical" ], atol=std_error_limit * t_h_c.std() / sqrt(len(t_h_c)))
        np.testing.assert_allclose(t_h_c_1.mean(), strain1_params[ "mean_time_to_critical" ], atol=std_error_limit * t_h_c_1.std() / sqrt(len(t_h_c_1)))

        # time from symptoms to recover if not hospitalised
        df_trans["t_s_r"] = df_trans["time_recovered"] - df_trans["time_symptomatic"]
        df_t_s_r = df_trans[(df_trans["time_recovered"] > 0) & (df_trans["time_asymptomatic"] < 0) & (df_trans["time_hospitalised"] <  0)]
        t_s_r   = df_t_s_r[df_t_s_r["strain_idx"]==0]["t_s_r"]
        t_s_r_1 = df_t_s_r[df_t_s_r["strain_idx"]==1]["t_s_r"]
        np.testing.assert_allclose(t_s_r.mean(),  test_params[ "mean_time_to_recover" ],   atol=std_error_limit * t_s_r.std() / sqrt(len(t_s_r)))
        np.testing.assert_allclose(t_s_r.std(),   test_params[ "sd_time_to_recover" ],     atol=std_error_limit * t_s_r.std() / sqrt(len(t_s_r)))
        np.testing.assert_allclose(t_s_r_1.mean(),strain1_params[ "mean_time_to_recover" ],atol=std_error_limit * t_s_r_1.std() / sqrt(len(t_s_r_1)))
        np.testing.assert_allclose(t_s_r_1.std(), strain1_params[ "sd_time_to_recover" ],  atol=std_error_limit * t_s_r_1.std() / sqrt(len(t_s_r_1)))

        # time from hospitalised to recover if don't got to ICU
        df_trans["t_h_r"] = df_trans["time_recovered"] - df_trans["time_hospitalised"]
        df_t_h_r = df_trans[(df_trans["time_recovered"] > 0) & (df_trans["time_critical"] < 0) & (df_trans["time_hospitalised"] > 0)]
        t_h_r    = df_t_h_r[df_t_h_r["strain_idx"]==0]["t_h_r"]     
        t_h_r_1  = df_t_h_r[df_t_h_r["strain_idx"]==1]["t_h_r"]     
        np.testing.assert_allclose(t_h_r.mean(),  test_params[ "mean_time_hospitalised_recovery" ],   atol=std_error_limit * t_h_r.std() / sqrt(len(t_h_r)))
        np.testing.assert_allclose(t_h_r.std(),   test_params[ "sd_time_hospitalised_recovery" ],     atol=std_error_limit * t_h_r.std() / sqrt(len(t_h_r)))
        np.testing.assert_allclose(t_h_r_1.mean(),strain1_params[ "mean_time_hospitalised_recovery" ],atol=std_error_limit * t_h_r_1.std() / sqrt(len(t_h_r_1)))
        np.testing.assert_allclose(t_h_r_1.std(), strain1_params[ "sd_time_hospitalised_recovery" ],  atol=std_error_limit * t_h_r_1.std() / sqrt(len(t_h_r_1)))

        # time in ICU
        df_trans["t_c_r"] = df_trans["time_hospitalised_recovering"] - df_trans["time_critical"]
        df_t_c_r = df_trans[ (df_trans["time_hospitalised_recovering"] > 0) & (df_trans["time_critical"] > 0)]
        t_c_r    = df_t_c_r[df_t_c_r["strain_idx"]==0]["t_c_r"]
        t_c_r_1  = df_t_c_r[df_t_c_r["strain_idx"]==1]["t_c_r"]
        np.testing.assert_allclose(t_c_r.mean(),  test_params[ "mean_time_critical_survive" ],   atol=std_error_limit * t_c_r.std() / sqrt(len(t_c_r)))
        np.testing.assert_allclose(t_c_r.std(),   test_params[ "sd_time_critical_survive" ],     atol=std_error_limit * t_c_r.std() / sqrt(len(t_c_r)))
        np.testing.assert_allclose(t_c_r_1.mean(),strain1_params[ "mean_time_critical_survive" ],atol=std_error_limit * t_c_r_1.std() / sqrt(len(t_c_r_1)))
        np.testing.assert_allclose(t_c_r_1.std(), strain1_params[ "sd_time_critical_survive" ],  atol=std_error_limit * t_c_r_1.std() / sqrt(len(t_c_r_1)))

        # time from ICU to death
        df_trans["t_c_d"] = df_trans["time_death"] - df_trans["time_critical"]
        df_t_c_d =  df_trans[(df_trans["time_death"] > 0) & (df_trans["time_critical"] > 0) ]
        t_c_d   = df_t_c_d[df_t_c_d["strain_idx"]==0]["t_c_d"]
        t_c_d_1 = df_t_c_d[df_t_c_d["strain_idx"]==1]["t_c_d"]
        np.testing.assert_allclose(t_c_d.mean(),  test_params[ "mean_time_to_death" ],   atol=std_error_limit * t_c_d.std() / sqrt(len(t_c_d)))
        np.testing.assert_allclose(t_c_d.std(),   test_params[ "sd_time_to_death" ],     atol=std_error_limit * t_c_d.std() / sqrt(len(t_c_d)))
        np.testing.assert_allclose(t_c_d_1.mean(),strain1_params[ "mean_time_to_death" ],atol=std_error_limit * t_c_d_1.std() / sqrt(len(t_c_d_1)))
        np.testing.assert_allclose(t_c_d_1.std(), strain1_params[ "sd_time_to_death" ],  atol=std_error_limit * t_c_d_1.std() / sqrt(len(t_c_d_1)))

        # time from asymptomatic to recover
        df_trans["t_a_r"] = df_trans["time_recovered"] - df_trans["time_asymptomatic"]
        df_t_a_r = df_trans[(df_trans["time_recovered"] > 0) & (df_trans["time_asymptomatic"] > 0)]
        t_a_r    = df_t_a_r[df_t_a_r["strain_idx"]==0]["t_a_r"]
        t_a_r_1  = df_t_a_r[df_t_a_r["strain_idx"]==1]["t_a_r"]
        np.testing.assert_allclose(t_a_r.mean(),  test_params[ "mean_asymptomatic_to_recovery" ],   atol=std_error_limit * t_a_r.std() / sqrt(len(t_a_r)))
        np.testing.assert_allclose(t_a_r.std(),   test_params[ "sd_asymptomatic_to_recovery" ],     atol=std_error_limit * t_a_r.std() / sqrt(len(t_a_r)))        
        np.testing.assert_allclose(t_a_r_1.mean(),strain1_params[ "mean_asymptomatic_to_recovery" ],atol=std_error_limit * t_a_r.std() / sqrt(len(t_a_r)))
        np.testing.assert_allclose(t_a_r_1.std(), strain1_params[ "sd_asymptomatic_to_recovery" ],  atol=std_error_limit * t_a_r.std() / sqrt(len(t_a_r)))
            
        
    def test_multi_strain_disease_outcome_proportions( self, test_params, strain1_params ):
        """
        Test that the fraction of infected people following each path for
        the progression of the disease agrees with the parameters (multi-strain version)
        """
        
        # set the np seed so the results are reproducible
        np.random.seed(0)      
        std_error_limit = 3
         
        # create model and add the new strain
        model = abm.Model( params = test_params )
        new_strain = model.add_new_strain( 1, 
            fraction_asymptomatic = strain1_params[ "fraction_asymptomatic" ],
            mild_fraction         = strain1_params[ "mild_fraction" ],
            hospitalised_fraction = strain1_params[ "hospitalised_fraction" ], 
            critical_fraction     = strain1_params[ "critical_fraction" ], 
            fatality_fraction     = strain1_params[ "fatality_fraction" ], 
        )
        seeds      = np.random.choice( test_params["n_total"], test_params["n_seed_infection"], replace=False)
        for idx in range( test_params["n_seed_infection"] ) :
            model.seed_infect_by_idx( seeds[ idx ], strain = new_strain )
            
        fraction_asymptomatic = [
            test_params[ "fraction_asymptomatic_0_9" ],
            test_params[ "fraction_asymptomatic_10_19" ],
            test_params[ "fraction_asymptomatic_20_29" ],
            test_params[ "fraction_asymptomatic_30_39" ],
            test_params[ "fraction_asymptomatic_40_49" ],
            test_params[ "fraction_asymptomatic_50_59" ],
            test_params[ "fraction_asymptomatic_60_69" ],
            test_params[ "fraction_asymptomatic_70_79" ],
            test_params[ "fraction_asymptomatic_80" ],
        ]
        fraction_asymptomatic_1 = strain1_params[ "fraction_asymptomatic" ]
        
        mild_fraction = [
            test_params[ "mild_fraction_0_9" ],
            test_params[ "mild_fraction_10_19" ],
            test_params[ "mild_fraction_20_29" ],
            test_params[ "mild_fraction_30_39" ],
            test_params[ "mild_fraction_40_49" ],
            test_params[ "mild_fraction_50_59" ],
            test_params[ "mild_fraction_60_69" ],
            test_params[ "mild_fraction_70_79" ],
            test_params[ "mild_fraction_80" ],
        ]
        mild_fraction_1 = strain1_params[ "mild_fraction" ]

        hospitalised_fraction = [
            test_params[ "hospitalised_fraction_0_9" ],
            test_params[ "hospitalised_fraction_10_19" ],
            test_params[ "hospitalised_fraction_20_29" ],
            test_params[ "hospitalised_fraction_30_39" ],
            test_params[ "hospitalised_fraction_40_49" ],
            test_params[ "hospitalised_fraction_50_59" ],
            test_params[ "hospitalised_fraction_60_69" ],
            test_params[ "hospitalised_fraction_70_79" ],
            test_params[ "hospitalised_fraction_80" ],
        ]
        hospitalised_fraction_1 = strain1_params[ "hospitalised_fraction"]

        critical_fraction = [
            test_params[ "critical_fraction_0_9" ],
            test_params[ "critical_fraction_10_19" ],
            test_params[ "critical_fraction_20_29" ],
            test_params[ "critical_fraction_30_39" ],
            test_params[ "critical_fraction_40_49" ],
            test_params[ "critical_fraction_50_59" ],
            test_params[ "critical_fraction_60_69" ],
            test_params[ "critical_fraction_70_79" ],
            test_params[ "critical_fraction_80" ],
        ]
        critical_fraction_1 = strain1_params[ "critical_fraction" ]

        fatality_fraction = [
            test_params[ "fatality_fraction_0_9" ],
            test_params[ "fatality_fraction_10_19" ],
            test_params[ "fatality_fraction_20_29" ],
            test_params[ "fatality_fraction_30_39" ],
            test_params[ "fatality_fraction_40_49" ],
            test_params[ "fatality_fraction_50_59" ],
            test_params[ "fatality_fraction_60_69" ],
            test_params[ "fatality_fraction_70_79" ],
            test_params[ "fatality_fraction_80" ],
        ]
        fatality_fraction_1 = strain1_params[ "fatality_fraction" ]

        model.run( verbose = False )
        df_trans = model.get_transmissions()
        df_indiv = model.get_individuals()
        df_indiv = pd.merge(df_indiv, df_trans, left_on = "ID", right_on = "ID_recipient", how = "left")

        # fraction asymptomatic vs mild+symptomatc
        df_inf      = df_indiv[ df_indiv["time_infected"] > 0 ]
        df_asym     = df_indiv[ df_indiv["time_asymptomatic"] > 0 ]
        df_mild_p   = df_indiv[ df_indiv["time_presymptomatic_mild"] > 0 ]
        df_sev_p    = df_indiv[ df_indiv["time_presymptomatic_severe"] > 0 ]
        df_sev      = df_indiv[ df_indiv["time_symptomatic_severe"] > 0 ]
        df_hosp     = df_indiv[ df_indiv["time_hospitalised"] > 0 ]
        df_crit     = df_indiv[ ( (df_indiv["time_critical"] > 0) | (df_indiv["time_death"] > 0) ) ]
        df_icu      = df_indiv[ ( (df_indiv["time_critical"] > 0) ) ]
        df_dead_icu = df_indiv[ ( (df_indiv["time_critical"] > 0) ) & (df_indiv["time_death"] > 0) ]
   
        for idx in range( constant.N_AGE_GROUPS ):

            N_inf        = len( df_inf[ ( df_inf["age_group"] == constant.AGES[idx] )       & ( df_inf["strain_idx"] == 0 ) ] )
            N_inf_1      = len( df_inf[ ( df_inf["age_group"] == constant.AGES[idx] )       & ( df_inf["strain_idx"] == 1 ) ] )
            N_asym       = len( df_asym[ ( df_asym["age_group"] == constant.AGES[idx] )     & ( df_asym["strain_idx"] == 0 ) ] )  
            N_asym_1     = len( df_asym[ ( df_asym["age_group"] == constant.AGES[idx] )     & ( df_asym["strain_idx"] == 1 ) ] )  
            N_sev_p      = len( df_sev_p[ ( df_sev_p["age_group"] == constant.AGES[idx] )   & ( df_sev_p["strain_idx"] == 0 ) ] )
            N_sev_p_1    = len( df_sev_p[ ( df_sev_p["age_group"] == constant.AGES[idx] )   & ( df_sev_p["strain_idx"] == 1 ) ] )
            N_mild_p     = len( df_mild_p[ ( df_mild_p["age_group"] == constant.AGES[idx] ) & ( df_mild_p["strain_idx"] == 0 ) ] )   
            N_mild_p_1   = len( df_mild_p[ ( df_mild_p["age_group"] == constant.AGES[idx] ) & ( df_mild_p["strain_idx"] == 1 ) ] )   
            N_sev        = len( df_sev[ ( df_sev["age_group"] == constant.AGES[idx] )       & ( df_sev["strain_idx"] == 0 ) ] )
            N_sev_1      = len( df_sev[ ( df_sev["age_group"] == constant.AGES[idx] )       & ( df_sev["strain_idx"] == 1 ) ] )
            N_hosp       = len( df_hosp[ ( df_hosp["age_group"] == constant.AGES[idx] )     & ( df_hosp["strain_idx"] == 0 ) ] )
            N_hosp_1     = len( df_hosp[ ( df_hosp["age_group"] == constant.AGES[idx] )     & ( df_hosp["strain_idx"] == 1 ) ] )
            N_crit       = len( df_crit[ ( df_crit["age_group"] == constant.AGES[idx] )     & ( df_crit["strain_idx"] == 0 ) ] )
            N_crit_1     = len( df_crit[ ( df_crit["age_group"] == constant.AGES[idx] )     & ( df_crit["strain_idx"] == 1 ) ] )
            N_icu        = len( df_icu[ ( df_icu["age_group"] == constant.AGES[idx] )       & ( df_icu["strain_idx"] == 0 ) ] )
            N_icu_1      = len( df_icu[ ( df_icu["age_group"] == constant.AGES[idx] )       & ( df_icu["strain_idx"] == 1 ) ] )
            N_dead_icu   = len( df_dead_icu[ ( df_dead_icu["age_group"] == constant.AGES[idx] ) & ( df_dead_icu["strain_idx"] == 0 ) ] )
            N_dead_icu_1 = len( df_dead_icu[ ( df_dead_icu["age_group"] == constant.AGES[idx] ) & ( df_dead_icu["strain_idx"] == 1 ) ] )
        
            np.testing.assert_( N_inf   == ( N_sev_p + N_asym + N_mild_p ),       msg = "missing infected people" )
            np.testing.assert_( N_inf_1 == ( N_sev_p_1 + N_asym_1 + N_mild_p_1 ), msg = "missing infected people" )
            np.testing.assert_( N_asym > 50,     msg = "insufficient asymptomtatic to test" )
            np.testing.assert_( N_asym_1 > 50,   msg = "insufficient asymptomtatic to test" )
            np.testing.assert_( N_mild_p > 50,   msg = "insufficient mild to test" )
            np.testing.assert_( N_mild_p_1 > 50, msg = "insufficient mild to test" )
            np.testing.assert_( N_hosp   > 50,   msg = "insufficient hospitalised to test" )
            np.testing.assert_( N_hosp_1 > 50,   msg = "insufficient hospitalised to test" )
            np.testing.assert_( N_icu    > 50,   msg = "insufficient ICU to test" )
            np.testing.assert_( N_icu_1  > 50,   msg = "insufficient ICU to test" )
            np.testing.assert_allclose( N_asym,     N_inf * fraction_asymptomatic[idx],    atol=std_error_limit * sqrt( N_inf * fraction_asymptomatic[idx ] ),  err_msg = "incorrect asymptomatics" )
            np.testing.assert_allclose( N_asym_1,   N_inf_1 * fraction_asymptomatic_1[idx],atol=std_error_limit * sqrt( N_inf_1 * fraction_asymptomatic_1[idx ] ),err_msg = "incorrect asymptomatics" )
            np.testing.assert_allclose( N_mild_p,   N_inf * mild_fraction[idx],            atol=std_error_limit * sqrt( N_inf * mild_fraction[idx] ),           err_msg = "incorrect milds" )        
            np.testing.assert_allclose( N_mild_p_1, N_inf_1 * mild_fraction_1[idx],        atol=std_error_limit * sqrt( N_inf_1 * mild_fraction_1[idx] ),         err_msg = "incorrect milds" )        
            np.testing.assert_allclose( N_hosp,     N_sev * hospitalised_fraction[idx],    atol=std_error_limit * sqrt( N_sev * hospitalised_fraction[idx] ),   err_msg = "incorrect hospitalised" )        
            np.testing.assert_allclose( N_hosp_1,   N_sev_1 * hospitalised_fraction_1[idx],atol=std_error_limit * sqrt( N_sev_1 * hospitalised_fraction_1[idx] ), err_msg = "incorrect hospitalised" )        
            np.testing.assert_allclose( N_crit,     N_hosp * critical_fraction[idx],       atol=std_error_limit * sqrt( N_hosp * critical_fraction[idx] ),    err_msg = "incorrect critical" )        
            np.testing.assert_allclose( N_crit_1,   N_hosp_1 * critical_fraction_1[idx],   atol=std_error_limit * sqrt( N_hosp_1 * critical_fraction_1[idx] ),    err_msg = "incorrect critical" )        
            np.testing.assert_allclose( N_dead_icu, N_icu * fatality_fraction[idx],        atol=std_error_limit * sqrt( N_icu * fatality_fraction[idx] ),     err_msg = "incorrect fatalitiy" )        
            np.testing.assert_allclose( N_dead_icu_1,N_icu_1 * fatality_fraction_1[idx],   atol=std_error_limit * sqrt( N_icu_1 * fatality_fraction_1[idx] ),     err_msg = "incorrect fatalitiy" )        
    
    def test_multi_strain_infectious_factor(self, test_params):
        """
        Test that the symptom type infectious factors for multiple strains
        Set the params so each strain can only be transmitted by sources with one of the 3 types
        and then check that only these transmissions take place
        
        """
        # set the np seed so the results are reproducible
        np.random.seed(0)
        std_error_limit = 4  
        eps = 0.00001
        
        # add a new strain and seed the infection
        model = abm.Model(params=test_params)
        strain_1 = model.add_new_strain( eps, mild_infectious_factor = 1 / eps )
        strain_2 = model.add_new_strain( eps, asymptomatic_infectious_factor = 1 / eps )
        seeds = np.random.choice(test_params["n_total"], 2 * test_params["n_seed_infection"], replace=False)
        for idx in range(test_params["n_seed_infection"]) :
            model.seed_infect_by_idx(seeds[ idx ], strain=strain_1)
            model.seed_infect_by_idx(seeds[ idx + test_params["n_seed_infection"] ], strain=strain_2)
        
        # run the model and look at the infections
        model.run(verbose=False)
        df_trans = model.get_transmissions()
        df_trans[ "symptom_type" ] = 0  + ( df_trans["time_presymptomatic_mild"] >= 0 ) + ( df_trans["time_presymptomatic_severe"] >= 0 ) * 2 
        df_source = df_trans.loc[ :,list(["ID_recipient", "symptom_type"])]
        df_source.rename( columns = {"ID_recipient":"ID_source"}, inplace = True)
        df_trans  = df_trans.loc[ :,list(["ID_source", "strain_idx","time_infected"])] 
        df_trans = df_trans[ list( df_trans[ "time_infected" ] > 1 ) ] # remove seeding events
        df_trans = pd.merge(df_trans, df_source, left_on="ID_source", right_on="ID_source", how="left")
        df_trans.symptom_type = df_trans.symptom_type.astype(int)
                
        np.testing.assert_( sum( ( df_trans["strain_idx"] == 0 ) & ( df_trans["symptom_type"] == 0 ) ) == 0,   msg = "asymptomatic infector with strain 0")
        np.testing.assert_( sum( ( df_trans["strain_idx"] == 0 ) & ( df_trans["symptom_type"] == 1 ) ) == 0,   msg = "mild infector with strain 0")
        np.testing.assert_( sum( ( df_trans["strain_idx"] == 0 ) & ( df_trans["symptom_type"] == 2 ) ) > 1000, msg = "insufficent severe infectors with strain 0")
        np.testing.assert_( sum( ( df_trans["strain_idx"] == 1 ) & ( df_trans["symptom_type"] == 0 ) ) == 0,   msg = "asymptomatic infector with strain 1")
        np.testing.assert_( sum( ( df_trans["strain_idx"] == 1 ) & ( df_trans["symptom_type"] == 1 ) ) > 1000, msg = "insufficient mild infector with strain 1")
        np.testing.assert_( sum( ( df_trans["strain_idx"] == 1 ) & ( df_trans["symptom_type"] == 2 ) ) == 0,   msg = "severe infector with strain 1")       
        np.testing.assert_( sum( ( df_trans["strain_idx"] == 2 ) & ( df_trans["symptom_type"] == 0 ) ) > 1000, msg = "insufficient asymptomatic infectors with strain 2")
        np.testing.assert_( sum( ( df_trans["strain_idx"] == 2 ) & ( df_trans["symptom_type"] == 1 ) ) == 0,   msg = "mild infector with strain 2")
        np.testing.assert_( sum( ( df_trans["strain_idx"] == 2 ) & ( df_trans["symptom_type"] == 2 ) ) == 0,   msg = "severe infectorswith strain 2")
