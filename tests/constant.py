#!/usr/bin/env python3
""""
constant.py 

constants used across all the testing files and
constants from the C code which are used in testing
(would be good to get this direct from C if possible)

 Created on: 31 Mar 2020
     Author: hinchr
"""

from os.path import join
from enum import Enum
import os

# Directories
IBM_DIR = "src"
IBM_DIR_TEST = "src_test"
DATA_DIR_TEST = "data_test"

TEST_DATA_TEMPLATE = "./tests/data/baseline_parameters.csv"
TEST_DATA_FILE = join(DATA_DIR_TEST, "test_parameters.csv")
TEST_HOSPITAL_FILE = "./tests/data/hospital_baseline_parameters.csv"

TEST_OUTPUT_FILE = join(DATA_DIR_TEST, "test_output.csv")
TEST_INDIVIDUAL_FILE = join(DATA_DIR_TEST, "individual_file_Run1.csv")
TEST_INTERACTION_FILE = join(DATA_DIR_TEST, "interactions_Run1.csv")
TEST_TRANSMISSION_FILE = join(DATA_DIR_TEST, "transmission_Run1.csv")
TEST_TRACE_FILE = join(DATA_DIR_TEST, "trace_tokens_Run1.csv")

TEST_HOUSEHOLD_TEMPLATE = "./tests/data/baseline_household_demographics.csv"
TEST_HOUSEHOLD_FILE = join(DATA_DIR_TEST, "test_household_demographics.csv")

class EVENT_TYPES(Enum):
    UNINFECTED = 0
    PRESYMPTOMATIC = 1
    PRESYMPTOMATIC_MILD = 2
    ASYMPTOMATIC = 3
    SYMPTOMATIC = 4
    SYMPTOMATIC_MILD = 5
    HOSPITALISED = 6
    CRITICAL = 7
    RECOVERED = 8
    DEATH = 9
    QUARANTINED = 10
    QUARANTINE_RELEASE = 11
    TEST_TAKE = 12
    TEST_RESULT = 13
    CASE = 14
    TRACE_TOKEN_RELEASE = 15
    NOT_IN_HOSPITAL = 16
    WAITING = 17
    GENERAL = 18
    ICU = 19
    MORTUARY = 20
    DISCHARGED = 21
    N_EVENT_TYPES = 22

# Age groups
AGE_0_9 = 0
AGE_10_19 = 1
AGE_20_29 = 2
AGE_30_39 = 3
AGE_40_49 = 4
AGE_50_59 = 5
AGE_60_69 = 6
AGE_70_79 = 7
AGE_80 = 8
N_AGE_GROUPS = 9
AGES = [
    AGE_0_9,
    AGE_10_19,
    AGE_20_29,
    AGE_30_39,
    AGE_40_49,
    AGE_50_59,
    AGE_60_69,
    AGE_70_79,
    AGE_80
]

CHILD = 0
ADULT = 1
ELDERLY = 2
AGE_TYPES = [CHILD, CHILD, ADULT, ADULT, ADULT, ADULT, ADULT, ELDERLY, ELDERLY]

# network type
HOUSEHOLD = 0
WORK = 1
RANDOM = 2
HOSPITAL_WORK = 3
HOSPITAL_DOCTOR_PATIENT_GENERAL = 4
HOSPITAL_NURSE_PATIENT_GENERAL = 5
HOSPITAL_DOCTOR_PATIENT_ICU = 6
HOSPITAL_NURSE_PATIENT_ICU = 7

# work networks
NETWORK_0_9 = 0
NETWORK_10_19 = 1
NETWORK_20_69 = 2
NETWORK_70_79 = 3
NETWORK_80 = 4
NETWORKS = [NETWORK_0_9, NETWORK_10_19, NETWORK_20_69, NETWORK_70_79, NETWORK_80]

# work type networks
NETWORK_CHILD = 0
NETWORK_ADULT = 1
NETWORK_ELDERLY = 2
NETWORK_TYPES = [NETWORK_CHILD, NETWORK_ADULT, NETWORK_ELDERLY]

PARAM_LINE_NUMBER = 1
HOSPITAL_PARAM_LINE_NUMBER = 1

# worker type
NOT_HEALTHCARE_WORKER = -1
DOCTOR = 0
NURSE = 1

# Construct the executable command
# EXE = f"covid19ibm.exe {TEST_DATA_FILE} {PARAM_LINE_NUMBER} "+\
#     f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE}"+\
#     f"{TEST_HOSPITAL_FILE} {HOSPITAL_PARAM_LINE_NUMBER}"

EXE = f"covid19ibm.exe {TEST_DATA_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {TEST_HOSPITAL_FILE}"

command = join(IBM_DIR_TEST, EXE)
