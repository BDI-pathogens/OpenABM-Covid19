#!/usr/bin/env python3
"""
Tests of the individual-based model, COVID19-IBM, using the individual file

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be 
run by calling 'pytest' from project folder.  

Created: March 2020
Author: p-robot
"""

import pytest, sys, subprocess, shutil, os
from os.path import join
import numpy as np, pandas as pd
from scipy import optimize
from math import sqrt, log, exp

sys.path.append("src/COVID19")
from parameters import ParameterSet

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
        "test_quarantine_interactions": [
            dict(
                test_params=dict(
                    n_total=50000,
                    quarantined_daily_interactions=0,
                    end_time=25,
                    infectious_rate=4,
                    self_quarantine_fraction=1.0,
                    seasonal_flu_rate=0.0,
                )
            ),
            dict(
                test_params=dict(
                    n_total=50000,
                    quarantined_daily_interactions=1,
                    end_time=25,
                    infectious_rate=4,
                    self_quarantine_fraction=0.75,
                    seasonal_flu_rate=0.0005,
                )
            ),
            dict(
                test_params=dict(
                    n_total=50000,
                    quarantined_daily_interactions=2,
                    end_time=25,
                    infectious_rate=4,
                    self_quarantine_fraction=0.50,
                    seasonal_flu_rate=0.001,
                )
            ),
            dict(
                test_params=dict(
                    n_total=50000,
                    quarantined_daily_interactions=3,
                    end_time=25,
                    infectious_rate=4,
                    self_quarantine_fraction=0.25,
                    seasonal_flu_rate=0.005,
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
                    seasonal_flu_rate=0.0,
                    asymptomatic_infectious_factor=0.4,
                )
            ),
            dict(
                test_params=dict(
                    n_total=50000,
                    end_time=1,
                    infectious_rate=4,
                    self_quarantine_fraction=0.5,
                    seasonal_flu_rate=0.05,
                    asymptomatic_infectious_factor=1.0,
                )
            ),
        ],
    }
    """
    Test class for checking 
    """
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
        df_indiv = pd.read_csv(
            constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True
        )
        df_int = pd.read_csv(
            constant.TEST_INTERACTION_FILE, comment="#", sep=",", skipinitialspace=True
        )

        # get the people who are in quarantine and were on the previous step
        df_quar = df_indiv[
            (df_indiv["quarantined"] == 1) & (df_indiv["time_quarantined"] < end_time)
        ]
        df_quar = df_quar.loc[:, "ID"]

        # get the number of interactions by type
        df_int = df_int.groupby(["ID", "type"]).size().reset_index(name="connections")

        # check to see there are no work connections
        df_test = pd.merge(
            df_quar, df_int[df_int["type"] == constant.WORK], on="ID", how="inner"
        )
        np.testing.assert_equal(
            len(df_test), 0, "quarantined individual with work contacts"
        )

        # check to see there are are household connections
        df_test = pd.merge(
            df_quar, df_int[df_int["type"] == constant.HOUSEHOLD], on="ID", how="inner"
        )
        np.testing.assert_equal(
            len(df_test) > 0,
            True,
            "quarantined individuals have no household connections",
        )

        # check to whether the number of random connections are as specified
        df_test = pd.merge(
            df_quar, df_int[df_int["type"] == constant.RANDOM], on="ID", how="left"
        )
        df_test.fillna(0, inplace=True)
        np.testing.assert_allclose(
            df_test.loc[:, "connections"].mean(),
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
        df_int = pd.read_csv(
            constant.TEST_INTERACTION_FILE, comment="#", sep=",", skipinitialspace=True
        )

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
        if test_params["seasonal_flu_rate"] == 0:
            df = pd.merge(df_quar, df_symp, on="ID", how="inner")
            np.testing.assert_equal(
                n_quar,
                len(df),
                "people quarantined without symptoms when seasonal flu turned off",
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
            n_flu = test_params["n_total"] * test_params["seasonal_flu_rate"]
            n_exp_quar = n_flu * test_params["self_quarantine_fraction"]

            np.testing.assert_allclose(
                n_exp_quar,
                n_quar,
                atol=tol_sd * sqrt(n_exp_quar),
                err_msg="the number of quarantined not explained by seasonal flu",
            )

        else:
            np.testing.assert_equal(
                True, False, "no test run due test_params not being testable"
            )
