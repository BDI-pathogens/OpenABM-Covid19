#!/usr/bin/env python3
"""
Tests of the individual-based model, COVID19-IBM

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be 
run by calling 'pytest' from project folder.  

Created: April 2020
Author: feldnerd
"""

import subprocess
import numpy as np, pandas as pd
import pytest

# File paths/ command line parameters
TEST_DATA_FILE = "/Users/feldnerd/Documents/GitHub/COVID19-IBM/tests/data/baseline_parameters.csv"
PARAM_LINE_NUMBER = 1
DATA_DIR_TEST = "/Users/feldnerd/Documents/GitHub/COVID19-IBM/tests/data"
TEST_HOUSEHOLD_FILE = "/Users/feldnerd/Documents/GitHub/COVID19-IBM/tests/data/baseline_household_demographics.csv"
TEST_HOSPITAL_FILE = "/Users/feldnerd/Documents/GitHub/COVID19-IBM/tests/data/hospital_baseline_parameters.csv"
TEST_OUTPUT_FILE = "/Users/feldnerd/Documents/GitHub/COVID19-IBM/tests/data/test_output.csv"
TEST_OUTPUT_FILE_HOSPITAL = "/Users/feldnerd/Documents/GitHub/COVID19-IBM/tests/data/test_hospital_output.csv"
TEST_INTERACTIONS_FILE = "/Users/feldnerd/Documents/GitHub/COVID19-IBM/tests/data/interactions_Run1.csv"
EXECUTABLE = "/Users/feldnerd/Documents/GitHub/COVID19-IBM/src/covid19ibm.exe"


# Construct the executable command
EXE = f"{EXECUTABLE} {TEST_DATA_FILE} {PARAM_LINE_NUMBER} "+\
    f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {TEST_HOSPITAL_FILE}"

# Call the model using baseline parameters, pipe output to file, read output file
file_output = open(TEST_OUTPUT_FILE, "w")
completed_run = subprocess.run([EXE], stdout = file_output, shell = True)


# Create a dataframe out of the terminal output
numHeader = 10
numFooter = 27
df_output = pd.read_csv(TEST_OUTPUT_FILE, 
    comment = "#", 
    sep = ",", 
    skiprows=numHeader, 
    skipfooter=numFooter, 
    engine='python')


# Write df_output to file
df_output.to_csv(TEST_OUTPUT_FILE_HOSPITAL, index = False)


class TestClass(object):
    """
    Test class for checking
    """

    def test_hospital_output(self):
        """
        Test that a dataframe of the correct size is being created
        """
        
        # Assert output not empty
        assert df_output.size > 1

        # output correct size (== time limit)
        r = len(df_output.index)
        assert r == 200


    def test_interactions(self):
        """
        Test that hospital workers are not in the list of work interac
        """
        # Load interactions df
        df_interactions = pd.read_csv(TEST_INTERACTIONS_FILE)

        # Check that interactions file was loaded df_interactions.columns
        columnArray = df_interactions.columns.values
        np.testing.assert_equal(columnArray, ["ID","age_group","worker_type_1","house_no","work_network","type","ID_2","age_group_2","worker_type_2","house_no_2","work_network_2"])

        # Check that none of the interactions occur with healthcare workers
        interactionCount = len(df_interactions.index)
        nonHealthCareWorkerType = -1

        # Array of individual types
        indivTypeList_1 = df_interactions.worker_type_1.values
        indivTypeList_2 = df_interactions.worker_type_2.values

        # Array of work_networks
        indivWorkerNetwork_1 = df_interactions.work_network.values
        indivWorkerNetwork_2 = df_interactions.work_network_2.values

        # Iterate over interaction arrays
        for ind in range(interactionCount):

            # If individuals are healthworkers, their work network is zero
            if(indivTypeList_1[ind] != nonHealthCareWorkerType):
                assert indivWorkerNetwork_1[ind] == 0
            if(indivTypeList_2[ind] != nonHealthCareWorkerType):
                assert indivWorkerNetwork_2[ind] == 0


        # This does the same as above, except much slower
        # # Iterate over dataframe, check whether healthcare workers have zero as their work network
        # for index, row in df_interactions.iterrows():

        #     worker_type_1 = row["worker_type_1"]
        #     worker_type_2 = row["worker_type_2"]

        #     work_network = row["work_network"]
        #     work_network_2 = row["work_network_2"]

        #     # Worker type healthworker should have work_network = 0
        #     if(worker_type_1 != nonHealthCareWorkerType):
        #         assert work_network == 0

        #     # Worker type healthworker should have work_network = 0
        #     if(worker_type_2 != nonHealthCareWorkerType):
        #         assert work_network_2 == 0





















