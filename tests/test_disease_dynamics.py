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
from math import sqrt

sys.path.append("src/COVID19")
from parameters import ParameterSet

from . import constant
from . import utilities as utils


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
    }
    """
    Test class for checking
    """
    def test_zero_recovery(self):
        """
        Setting recover times to be very large should avoid seeing any in recovered compartment
        """
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)

        # Make recovery very long
        params.set_param("mean_time_to_recover", 200.0)
        params.set_param("mean_asymptomatic_to_recovery", 200.0)
        params.set_param("mean_time_hospitalised_recovery", 200.0)

        params.write_params(constant.TEST_DATA_FILE)

        # Call the model
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")

        np.testing.assert_array_equal(
            df_output[["n_recovered"]].sum(),
            0)

    def test_zero_deaths(self):
        """
        Set fatality ratio to zero, should have no deaths if always places in the ICU
        """
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params = utils.set_fatality_fraction_all(params, 0.0)
        params = utils.set_location_death_icu_all(params, 1.0)
        params.write_params(constant.TEST_DATA_FILE)

        # Call the model, pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")

        np.testing.assert_equal(df_output["n_death"].sum(), 0)

    def test_total_infectious_rate_zero(self):
        """
        Set infectious rate to zero results in only "n_seed_infection" as total_infected
        """
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params.set_param("infectious_rate", 0.0)
        params.write_params(constant.TEST_DATA_FILE)

        # Call the model, pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)

        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")

        output = df_output["total_infected"].iloc[-1]
        expected_output = int(params.get_param("n_seed_infection"))

        np.testing.assert_equal(output, expected_output)

    def test_zero_infected(self):
        """
        Set seed-cases to zero should result in zero sum of output column
        """
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params.set_param("n_seed_infection", 0)
        params.write_params(constant.TEST_DATA_FILE)

        # Call the model, pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")

        np.testing.assert_equal(df_output["total_infected"].sum(), 0)

    def test_disease_transition_times( self, test_params ):
        """
        Test that the mean and standard deviation of the transition times between
        states agrees with the parameters
        """
        std_error_limit = 4

        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params = utils.turn_off_interventions(params, 50)
        params.set_param("n_total", 20000)
        params.set_param("n_seed_infection", 200)
        params.set_param("end_time", 50)
        params.set_param("infectious_rate", 4.0)
        params.set_param("hospital_on", 0) #turning off hospital as this affects disease transitions
        params.set_param( test_params )

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

        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params = utils.turn_off_interventions(params, 50)

        params.set_param("n_total", 20000)
        params.set_param("n_seed_infection", 200)
        params.set_param("end_time", 250)
        params.set_param("infectious_rate", 4.0)
        params.set_param("mild_infectious_factor", 1.0)
        params.set_param("hospital_on", 0)
        params.set_param( test_params )

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
