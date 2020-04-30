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

# Directories
IBM_DIR = "src"
IBM_DIR_TEST = "src_test"
DATA_DIR_TEST = "data_test"

TEST_DATA_TEMPLATE = "./tests/data/baseline_parameters.csv"
TEST_DATA_FILE = join(DATA_DIR_TEST, "test_parameters.csv")

TEST_OUTPUT_FILE = join(DATA_DIR_TEST, "test_output.csv")
TEST_INDIVIDUAL_FILE = join(DATA_DIR_TEST, "individual_file_Run1.csv")
TEST_INTERACTION_FILE = join(DATA_DIR_TEST, "interactions_Run1.csv")
TEST_TRANSMISSION_FILE = join(DATA_DIR_TEST, "transmission_Run1.csv")
TEST_TRACE_FILE = join(DATA_DIR_TEST, "trace_tokens_Run1.csv")

TEST_HOUSEHOLD_TEMPLATE = "./tests/data/baseline_household_demographics.csv"
TEST_HOUSEHOLD_FILE = join(DATA_DIR_TEST, "test_household_demographics.csv")

class EVENT_TYPES(Enum):
    SUSCEPTIBLE = 0
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
    N_EVENT_TYPES = 16

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

# work networks
PRIMARY_NETWORK = 0
SECONDARY_NETWORK = 1
WORKING_NETWORK = 2
RETIRED_NETWORK = 3
ELDERLY_NETWORK = 4
NETWORKS = [PRIMARY_NETWORK, SECONDARY_NETWORK, WORKING_NETWORK, RETIRED_NETWORK, ELDERLY_NETWORK]

# work type networks
NETWORK_CHILD = 0
NETWORK_ADULT = 1
NETWORK_ELDERLY = 2
NETWORK_TYPES = [NETWORK_CHILD, NETWORK_ADULT, NETWORK_ELDERLY]

MAX_DAILY_INTERACTIONS_KEPT = 10

PARAM_LINE_NUMBER = 1

# Construct the executable command
EXE = f"covid19ibm.exe {TEST_DATA_FILE} {PARAM_LINE_NUMBER} "+\
    f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE}"

command = join(IBM_DIR_TEST, EXE)
