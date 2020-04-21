#!/usr/bin/env python3
"""
Tests of the individual-based model, COVID19-IBM

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be 
run by calling 'pytest' from project folder.  

Created: April 2020
Author: Dylan Feldner-Busztin
"""

import subprocess, pytest, os, sys
import numpy as np, pandas as pd

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_HOSPITAL_FILE = TEST_DIR +"/data/hospital_baseline_parameters.csv"
PYTHON_C_DIR = TEST_DIR.replace("tests","") + "src/COVID19"
SCENARIO_HOSPITAL_FILE = TEST_DIR + "/data/new_hospital_baseline_parameters.csv"

sys.path.append(PYTHON_C_DIR)
from parameters import ParameterSet

params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)

print(params.list_params())
print(params.get_param("sd_time_hospital_transition"))
params.set_param("sd_time_hospital_transition", 4)

params.write_params(SCENARIO_HOSPITAL_FILE)


