#!/usr/bin/env python3
"""
Tests file concordance across the different filetypes of the COVID19-IBM

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
        df_timeseries = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
        
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        
        incidence_indiv, bins = np.histogram(df_indiv[(df_indiv[indiv_var] > 0)][indiv_var], 
            bins = np.arange(2, df_timeseries.time.max() + 1))
        
        incidence_timeseries = np.diff(df_timeseries[timeseries_var].values)
        
        np.testing.assert_array_equal(incidence_indiv, incidence_timeseries[:-1])
        
