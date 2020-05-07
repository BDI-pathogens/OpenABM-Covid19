#!/usr/bin/env python3
"""
Running the individual-based model, COVID19-IBM

Usage:
python3 run_simulation.py  

Created: May 2020
Author: Dylan Feldner-Busztin
"""

import subprocess, pytest, os, sys
import numpy as np, pandas as pd
from . import constant

# PYTHON_DIR = os.path.dirname(os.path.realpath(__file__))
# TEST_DATA_DIR = PYTHON_DIR.replace("python","") + "tests/data"
# TEST_DATA_FILE = TEST_DATA_DIR + "/baseline_parameters.csv"
# PARAM_LINE_NUMBER = 1
# TEST_HOUSEHOLD_FILE = TEST_DATA_DIR + "/baseline_household_demographics.csv"
# TEST_HOSPITAL_FILE = TEST_DATA_DIR +"/hospital_baseline_parameters.csv"
# TEST_OUTPUT_FILE = TEST_DATA_DIR +"/test_output.csv"
# TEST_OUTPUT_FILE_HOSPITAL = TEST_DATA_DIR +"/test_hospital_output.csv"
# TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP = TEST_DATA_DIR +"/time_step_hospital_output.csv"
# TEST_INTERACTIONS_FILE = TEST_DATA_DIR +"/interactions_Run1.csv"
# TEST_INDIVIDUAL_FILE = TEST_DATA_DIR +"/individual_file_Run1.csv"
# TEST_HCW_FILE = TEST_DATA_DIR +"/ward_output.csv"
# SRC_DIR = PYTHON_DIR.replace("python","") + "src"
# EXECUTABLE = SRC_DIR + "/covid19ibm.exe"


# # Construct the compilation command and compile
# compile_command = "make clean; make all; make swig-all;"
# completed_compilation = subprocess.run([compile_command], 
#     shell = True, 
#     cwd = SRC_DIR, 
#     capture_output = True
#     )

# # Construct the executable command
# EXE = f"{EXECUTABLE} {TEST_DATA_FILE} {PARAM_LINE_NUMBER} "+\
#     f"{TEST_DATA_DIR} {TEST_HOUSEHOLD_FILE} {TEST_HOSPITAL_FILE}"

# # Call the model using baseline parameters, pipe output to file, read output file
# file_output = open(TEST_OUTPUT_FILE, "w")
# completed_run = subprocess.run([EXE], stdout = file_output, shell = True)

print(constant.EVENT_TYPES.ICU)