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
from string import Template

# Directories
IBM_DIR = "src"
IBM_DIR_TEST = "src_test"
DATA_DIR_TEST = "data_test"

TEST_DATA_TEMPLATE = "./tests/data/baseline_parameters.csv"
TEST_DATA_FILE = join(DATA_DIR_TEST, "test_parameters.csv")

TEST_HOSPITAL_TEMPLATE = "./tests/data/hospital_baseline_parameters.csv"
TEST_HOSPITAL_FILE = join(DATA_DIR_TEST, "test_hospital_parameters.csv")

TEST_OUTPUT_FILE = join(DATA_DIR_TEST, "test_output.csv")
TEST_INDIVIDUAL_FILE = join(DATA_DIR_TEST, "individual_file_Run1.csv")
TEST_INTERACTION_FILE = join(DATA_DIR_TEST, "interactions_Run1.csv")
TEST_TRANSMISSION_FILE = join(DATA_DIR_TEST, "transmission_Run1.csv")
TEST_TRACE_FILE = join(DATA_DIR_TEST, "trace_tokens_Run1.csv")
TEST_QUARANTINE_REASONS_FILE = Template(join(DATA_DIR_TEST, "quarantine_reasons_file_Run1_T$T.csv"))
TEST_HCW_FILE = join(DATA_DIR_TEST, "ward_output1.csv")
TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP = join(DATA_DIR_TEST, "time_step_hospital_output1.csv")
TEST_OUTPUT_FILE_HOSPITAL_INTERACTIONS = join(DATA_DIR_TEST, "time_step_hospital_interactions1.csv")
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
    HOSPITALISED_RECOVERING = 8
    RECOVERED = 9
    RECOVERED_SUSCEPTIBLE = 10
    DEATH = 11
    QUARANTINED = 12
    QUARANTINE_RELEASE = 13
    TEST_TAKE = 14
    TEST_RESULT = 15
    CASE = 16
    TRACE_TOKEN_RELEASE = 17
    NOT_IN_HOSPITAL = 18
    WAITING = 19
    GENERAL = 20
    ICU = 21
    MORTUARY = 22
    DISCHARGED = 23
    N_EVENT_TYPES = 24

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
OCCUPATION = 1
RANDOM = 2
HOSPITAL_WORK = 3
HOSPITAL_DOCTOR_PATIENT_GENERAL = 4
HOSPITAL_NURSE_PATIENT_GENERAL = 5
HOSPITAL_DOCTOR_PATIENT_ICU = 6
HOSPITAL_NURSE_PATIENT_ICU = 7

# work networks
HOSPITAL_WORK_NETWORK = -1
PRIMARY_NETWORK = 0
SECONDARY_NETWORK = 1
WORKING_NETWORK = 2
RETIRED_NETWORK = 3
ELDERLY_NETWORK = 4
N_DEFAULT_OCCUPATION_NETWORKS = 5
NETWORKS = [PRIMARY_NETWORK, SECONDARY_NETWORK, WORKING_NETWORK, RETIRED_NETWORK, ELDERLY_NETWORK]

# work type networks
NETWORK_CHILD = 0
NETWORK_ADULT = 1
NETWORK_ELDERLY = 2
NETWORK_TYPES = [NETWORK_CHILD, NETWORK_ADULT, NETWORK_ELDERLY]

# network type map
NETWORK_TYPE_MAP = [
    NETWORK_CHILD,
    NETWORK_CHILD,
    NETWORK_ADULT,
    NETWORK_ELDERLY,
    NETWORK_ELDERLY
]

# custom network type map
CUSTOM_NETWORK_TYPE_MAP = [
    NETWORK_CHILD,
    NETWORK_CHILD,
    NETWORK_ADULT,
    NETWORK_ADULT,
    NETWORK_ADULT,
    NETWORK_ADULT,
    NETWORK_ADULT,
    NETWORK_ADULT,
    NETWORK_ELDERLY,
    NETWORK_ELDERLY
]

MAX_DAILY_INTERACTIONS_KEPT = 10

PARAM_LINE_NUMBER = 1
HOSPITAL_PARAM_LINE_NUMBER = 1

NOT_HEALTHCARE_WORKER = -1
class HCW_TYPES(Enum):
    DOCTOR = 0
    NURSE = 1

class HOSPITAL_WARD_TYPES(Enum):
    COVID_GENERAL = 0
    COVID_ICU = 1

# Construct the executable command
EXE = f"covid19ibm.exe {TEST_DATA_FILE} {PARAM_LINE_NUMBER} "+\
    f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {TEST_HOSPITAL_FILE}"

command = join(IBM_DIR_TEST, EXE)
