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
TEST_HOSPITAL_FILE = TEST_DIR + "/data/hospital_baseline_parameters.csv"
TEST_DATA_FILE = TEST_DIR + "/data/baseline_parameters.csv"
PARAM_LINE_NUMBER = 1
DATA_DIR_TEST = TEST_DIR + "/data"
TEST_HOUSEHOLD_FILE = TEST_DIR + "/data/baseline_household_demographics.csv"
TEST_OUTPUT_FILE = TEST_DIR + "/data/test_output.csv"
TEST_OUTPUT_FILE_HOSPITAL = TEST_DIR + "/data/test_hospital_output.csv"
TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP = TEST_DIR + "/data/time_step_hospital_output.csv"
TEST_INTERACTIONS_FILE = TEST_DIR + "/data/interactions_Run1.csv"
TEST_INDIVIDUAL_FILE = TEST_DIR + "/data/individual_file_Run1.csv"
TEST_HCW_FILE = TEST_DIR + "/data/ward_output.csv"
SRC_DIR = TEST_DIR.replace("tests", "") + "src"
EXECUTABLE = SRC_DIR + "/covid19ibm.exe"

# Use parameter file from Python C interface to adjust parameters
PYTHON_C_DIR = TEST_DIR.replace("tests","") + "src/COVID19"
sys.path.append(PYTHON_C_DIR)
from parameters import ParameterSet
from . import constant

# Files with adjusted parameters for each scenario
SCENARIO_FILE = TEST_DIR + "/data/scenario_baseline_parameters.csv"
SCENARIO_HOSPITAL_FILE = TEST_DIR + "/data/scenario_hospital_baseline_parameters.csv"


class TestClass(object):
    """
    Test class for checking
    """
# Dylan update for file change
    def test_zero_infectivity(self):
        """
        Set infections rate to zero, total infections should be equal to seed infections
        """

        # Adjust baseline parameter
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("infectious_rate", 0.0)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)

        # Construct the compilation command and compile
        # compile_command = "make clean; make all"
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command],
            shell = True,
            cwd = SRC_DIR,
            capture_output = True
            )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {TEST_HOSPITAL_FILE}"

        # Call the model using baseline parameters, pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        # In the individual file, time_infected should be not equal to -1 in n_seed_infection number of cases
        df_individual_output = pd.read_csv(TEST_INDIVIDUAL_FILE)

        expected_output = int(params.get_param("n_seed_infection"))
        output = df_individual_output["time_infected"] != -1
        output = df_individual_output[output]
        output = len(output.index)

        np.testing.assert_equal(output, expected_output)

# Dylan update for file change 
    def test_zero_beds(self):
        """
        Set hospital beds to zero
        """

        # Adjust baseline parameter
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_beds_covid_general_ward", 0)
        h_params.set_param("n_beds_covid_icu_ward", 0)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)

        # Construct the compilation command and compile
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command],
            shell = True,
            cwd = SRC_DIR,
            capture_output = True
            )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        df_time_step = pd.read_csv(TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

        n_patient_general = df_time_step["hospital_state"] == constant.EVENT_TYPES.GENERAL.value
        n_patient_general = df_time_step[n_patient_general]
        n_patient_general = len(n_patient_general.index)

        assert n_patient_general == 0

        n_patient_icu = df_time_step["hospital_state"] == constant.EVENT_TYPES.ICU.value
        n_patient_icu = df_time_step[n_patient_icu]
        n_patient_icu = len(n_patient_icu.index)

        assert n_patient_icu == 0

    def test_zero_general_wards(self):
        """
        Set hospital general wards to zero, check there are no general patients
        """

        # Adjust baseline parameter
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_covid_general_wards", 0)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)

        # Construct the compilation command and compile
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command],
            shell = True,
            cwd = SRC_DIR,
            capture_output = True
            )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        df_time_step = pd.read_csv(TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

        n_patient_general = df_time_step["hospital_state"] == constant.EVENT_TYPES.GENERAL.value
        n_patient_general = df_time_step[n_patient_icu]
        n_patient_general = len(n_patient_icu.index)

        assert n_patient_general == 0


    def test_zero_icu_wards(self):
        """
        Set hospital icu wards to zero, check there are no icu patients
        """

        # Adjust baseline parameter
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_covid_icu_wards", 0)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)

        # Construct the compilation command and compile
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command],
            shell = True,
            cwd = SRC_DIR,
            capture_output = True
            )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        df_time_step = pd.read_csv(TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

        n_patient_icu = df_time_step["hospital_state"] == constant.EVENT_TYPES.ICU.value
        n_patient_icu = df_time_step[n_patient_icu]
        n_patient_icu = len(n_patient_icu.index)

        assert n_patient_icu == 0


    def test_zero_hcw_patient_interactions(self):
        """
        Set patient required hcw interactions to zero
        and check that there are no interactions between
        hcw and patients
        """
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_patient_doctor_required_interactions_covid_general", 0)
        h_params.set_param("n_patient_nurse_required_interactions_covid_general_ward", 0)
        h_params.set_param("n_patient_doctor_required_interactions_covid_icu_ward", 0)
        h_params.set_param("n_patient_nurse_required_interactions_covid_icu_ward", 0)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command],
                                               shell=True,
                                               cwd=SRC_DIR,
                                               capture_output=True
                                               )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} " + \
              f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout=file_output, shell=True)
        df_interactions = pd.read_csv(TEST_INTERACTIONS_FILE,
                                      comment="#", sep=",", skipinitialspace=True)

        df_doctor_patient_general_interactions = df_interactions[
            df_interactions["type"] == constant.HOSPITAL_DOCTOR_PATIENT_GENERAL]
        df_nurse_patient_general_interactions = df_interactions[
            df_interactions["type"] == constant.HOSPITAL_NURSE_PATIENT_GENERAL]
        df_doctor_patient_icu_interactions = df_interactions[
            df_interactions["type"] == constant.HOSPITAL_DOCTOR_PATIENT_ICU]
        df_nurse_patient_icu_interactions = df_interactions[
            df_interactions["type"] == constant.HOSPITAL_NURSE_PATIENT_ICU]

        assert len(df_doctor_patient_general_interactions) == 0
        assert len(df_nurse_patient_general_interactions) == 0
        assert len(df_doctor_patient_icu_interactions) == 0
        assert len(df_nurse_patient_icu_interactions) == 0


    def test_hospital_waiting_modifiers(self):
        """
        Set patient ward beds to zero so that all patient enter a waiting state
        and set the waiting modifiers to zero and check that all patients recover
        """
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)
        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("hospitalised_waiting_mod", 0)
        h_params.set_param("critical_waiting_mod", 0)
        h_params.set_param("n_beds_covid_general_ward", 0)
        h_params.set_param("n_beds_covid_icu_ward", 0)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)
        # Construct the compilation command and compile
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command],
                                               shell=True,
                                               cwd=SRC_DIR,
                                               capture_output=True
                                               )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} " + \
              f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout=file_output, shell=True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        df_individual_output = pd.read_csv(TEST_INDIVIDUAL_FILE)
        n_deaths = df_individual_output["time_death"] != -1
        n_deaths = df_individual_output[n_deaths]
        assert len(n_deaths.index) == 0

        n_infected = df_individual_output["time_infected"] != -1
        n_recovered = df_individual_output["time_recovered"] != -1
        assert len(df_individual_output[n_infected].index) == len(df_individual_output[n_recovered].index)

# Dylan update for file change
    def test_hospital_infectivity_modifiers_zero(self):
        """
        Set patient infectivity modifiers to zero and test that no healthcare
        workers are infected by a patient
        """
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)
        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("waiting_infectivity_modifier", 0)
        h_params.set_param("general_infectivity_modifier", 0)
        h_params.set_param("icu_infectivity_modifier", 0)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)
        # Construct the compilation command and compile
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command],
                                               shell=True,
                                               cwd=SRC_DIR,
                                               capture_output=True
                                               )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} " + \
              f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout=file_output, shell=True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        df_individual_output = pd.read_csv(TEST_INDIVIDUAL_FILE)
        # get healthcare workers
        healthcare_workers = df_individual_output["worker_type"] != constant.NOT_HEALTHCARE_WORKER
        healthcare_workers = df_individual_output[healthcare_workers]

        # check that no healthcare workers have been infected by a patient
        for index, healthcare_worker in healthcare_workers.iterrows():
            infector_hospital_state = healthcare_worker["infector_hospital_state"]
            assert int(infector_hospital_state) != constant.EVENT_TYPES.WAITING.value
            assert int(infector_hospital_state) != constant.EVENT_TYPES.GENERAL.value
            assert int(infector_hospital_state) != constant.EVENT_TYPES.ICU.value

# Dylan update for file change
    def test_hospital_infectivity_modifiers_max(self):
        """
        Set patient infectivity modifiers to 100 and test that all healthcare
        workers who have interacted with a patient get infected
        """
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)
        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("waiting_infectivity_modifier", 100)
        h_params.set_param("general_infectivity_modifier", 100)
        h_params.set_param("icu_infectivity_modifier", 100)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)
        # Construct the compilation command and compile
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command],
                                               shell=True,
                                               cwd=SRC_DIR,
                                               capture_output=True
                                               )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} " + \
              f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout=file_output, shell=True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")


        # Check that the simulation ran
        assert len(df_output) != 0

        df_individual_output = pd.read_csv(TEST_INDIVIDUAL_FILE)
        df_int = pd.read_csv(TEST_INTERACTIONS_FILE, comment="#", sep=",", skipinitialspace=True)

        #get all healthcare workers who have interacted with patients
        hcw_with_patient_interaction = (df_int["worker_type_1"] != constant.NOT_HEALTHCARE_WORKER) & (df_int["type"] > constant.HOSPITAL_WORK) & (df_int["type"] <= constant.HOSPITAL_NURSE_PATIENT_ICU)
        hcw_with_patient_interaction = df_int[hcw_with_patient_interaction]
        # get healthcare workers
        for index, row in hcw_with_patient_interaction.iterrows():
            hcw_output = df_individual_output["ID"] == row["ID"]
            hcw_output = df_individual_output[hcw_output]
            assert hcw_output["time_infected"].value > -1


    def test_max_hcw_daily_interactions(self):
        """
        Set healthcare workers max daily interactions (with patients) to 0 and
        check that there are no disease transition from patient to healthcare workers
        or any interactions
        """
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)
        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("max_hcw_daily_interactions", 0)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)
        # Construct the compilation command and compile
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command],
                                               shell=True,
                                               cwd=SRC_DIR,
                                               capture_output=True
                                               )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} " + \
              f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout=file_output, shell=True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        df_individual_output = pd.read_csv(TEST_INDIVIDUAL_FILE)
        # get healthcare workers
        healthcare_workers = df_individual_output["worker_type"] != constant.NOT_HEALTHCARE_WORKER
        healthcare_workers = df_individual_output[healthcare_workers]

        # check that no healthcare workers have been infected by a patient
        for index, healthcare_worker in healthcare_workers.iterrows():
            infector_hospital_state = healthcare_worker["infector_hospital_state"]
            assert int(infector_hospital_state) != constant.EVENT_TYPES.WAITING.value
            assert int(infector_hospital_state) != constant.EVENT_TYPES.GENERAL.value
            assert int(infector_hospital_state) != constant.EVENT_TYPES.ICU.value

        df_interactions = pd.read_csv(TEST_INTERACTIONS_FILE,
                                      comment="#", sep=",", skipinitialspace=True)

        df_doctor_patient_general_interactions = df_interactions[
            df_interactions["type"] == constant.HOSPITAL_DOCTOR_PATIENT_GENERAL]
        df_nurse_patient_general_interactions = df_interactions[
            df_interactions["type"] == constant.HOSPITAL_NURSE_PATIENT_GENERAL]
        df_doctor_patient_icu_interactions = df_interactions[
            df_interactions["type"] == constant.HOSPITAL_DOCTOR_PATIENT_ICU]
        df_nurse_patient_icu_interactions = df_interactions[
            df_interactions["type"] == constant.HOSPITAL_NURSE_PATIENT_ICU]

        assert len(df_doctor_patient_general_interactions) == 0
        assert len(df_nurse_patient_general_interactions) == 0
        assert len(df_doctor_patient_icu_interactions) == 0
        assert len(df_nurse_patient_icu_interactions) == 0


    def test_zero_hospitalised_waiting_mod(self):
        """
        Set hospitalised_waiting_mod to zero, everybody
        in waiting state should be waiting or recovered in
        the next time steps
        """

        # Adjust baseline parameter
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("hospitalised_waiting_mod", 0.0)
        h_params.set_param("n_beds_covid_general_ward", 0)
        h_params.set_param("n_beds_covid_icu_ward", 0)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)

        # Construct the compilation command and compile
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command],
            shell = True,
            cwd = SRC_DIR,
            capture_output = True
            )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        time_step_df = pd.read_csv(TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

        for index, row in time_step_df.iterrows():

            if(row['hospital_state'] == constant.EVENT_TYPES.WAITING.value):

                ID = row['pdx']
                current_time_step = row['time_step']
                current_hospital_status = row['hospital_state']

                # Sub df with remaining timesteps for this individual
                rest_time_steps_condition = time_step_df['time_step'] > current_time_step
                pdx_condition = time_step_df['pdx'] == ID
                rest_time_steps_pdx_df = time_step_df[rest_time_steps_condition & pdx_condition]

                # Sub df with hospital_state either waiting or discharged
                waiting_condition = rest_time_steps_pdx_df['hospital_state'] == constant.EVENT_TYPES.WAITING.value
                discharged_condition = rest_time_steps_pdx_df['hospital_state'] == constant.EVENT_TYPES.DISCHARGED.value
                waiting_or_discharged_df = rest_time_steps_pdx_df[waiting_condition | discharged_condition]

                assert len(waiting_or_discharged_df.index) == len(rest_time_steps_pdx_df.index)


    def test_zero_critical_waiting_mod(self):
        """
        Set critical_waiting_mod to zero, everybody
        in waiting state should be waiting or recovered in
        the next time steps
        """

        # Adjust baseline parameter
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("critical_waiting_mod", 0.0)
        h_params.set_param("n_beds_covid_general_ward", 0)
        h_params.set_param("n_beds_covid_icu_ward", 0)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)

        # Construct the compilation command and compile
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command],
            shell = True,
            cwd = SRC_DIR,
            capture_output = True
            )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        time_step_df = pd.read_csv(TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

        for index, row in time_step_df.iterrows():


            if(row['hospital_state'] == constant.EVENT_TYPES.WAITING.value):

                ID = row['pdx']
                current_time_step = row['time_step']
                current_hospital_status = row['hospital_state']

                # Sub df with remaining timesteps for this individual
                rest_time_steps_condition = time_step_df['time_step'] > current_time_step
                pdx_condition = time_step_df['pdx'] == ID
                rest_time_steps_pdx_df = time_step_df[rest_time_steps_condition & pdx_condition]

                # Sub df with hospital_state either waiting or discharged
                waiting_condition = rest_time_steps_pdx_df['hospital_state'] == constant.EVENT_TYPES.WAITING.value
                discharged_condition = rest_time_steps_pdx_df['hospital_state'] == constant.EVENT_TYPES.DISCHARGED.value
                waiting_or_discharged_df = rest_time_steps_pdx_df[waiting_condition | discharged_condition]

                assert len(waiting_or_discharged_df.index) == len(rest_time_steps_pdx_df.index)


    def test_100_waiting_mods(self):
        """
        Set hospitalised_waiting_mod and critical_waiting
        to zero, everybody in waiting state should be in
        waiting, critical or mortuary in the next time steps
        """

        # Adjust baseline parameter
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("hospitalised_waiting_mod", 100.0)
        h_params.set_param("critical_waiting_mod", 100.0)
        h_params.set_param("n_beds_covid_general_ward", 0)
        h_params.set_param("n_beds_covid_icu_ward", 0)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)

        # Construct the compilation command and compile
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command],
            shell = True,
            cwd = SRC_DIR,
            capture_output = True
            )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        time_step_df = pd.read_csv(TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

        for index, row in time_step_df.iterrows():

            if(row['hospital_state'] == constant.EVENT_TYPES.WAITING.value):

                ID = row['pdx']
                current_time_step = row['time_step']
                current_hospital_status = row['hospital_state']

                # Sub df with remaining timesteps for this individual
                rest_time_steps_condition = time_step_df['time_step'] > current_time_step
                pdx_condition = time_step_df['pdx'] == ID
                rest_time_steps_pdx_df = time_step_df[rest_time_steps_condition & pdx_condition]

                # Sub df with hospital_state either waiting or discharged
                waiting_condition = rest_time_steps_pdx_df['hospital_state'] == constant.EVENT_TYPES.WAITING.value
                icu_condition = rest_time_steps_pdx_df['hospital_state'] == constant.EVENT_TYPES.ICU.value
                mortuary_condition = rest_time_steps_pdx_df['hospital_state'] == constant.EVENT_TYPES.MORTUARY.value
                waiting_or_discharged_df = rest_time_steps_pdx_df[waiting_condition | icu_condition | mortuary_condition]

                assert len(waiting_or_discharged_df.index) == len(rest_time_steps_pdx_df.index)


    def test_no_space_limit_beds(self):
        """
        Set number of beds in each ward to the population size,
        check that nobody in waiting state
        """

        # Adjust baseline parameter
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_beds_covid_general_ward", 20000)
        h_params.set_param("n_beds_covid_icu_ward", 20000)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)

        # Construct the compilation command and compile
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command],
            shell = True,
            cwd = SRC_DIR,
            capture_output = True
            )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        time_step_df = pd.read_csv(TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

        waiting_df = time_step_df['hospital_state'] == constant.EVENT_TYPES.WAITING.value
        waiting_df = time_step_df[waiting_df]

        assert len(waiting_df.index) == 0


    # def test_no_space_limit_wards(self):
    #     """
    #     Set number of wards in each ward to the population size,
    #     check that nobody in waiting state
    #     """

    #     # Adjust baseline parameter
    #     params = ParameterSet(TEST_DATA_FILE, line_number=1)
    #     params.set_param("n_total", 20000)
    #     params.write_params(SCENARIO_FILE)

    #     # Adjust hospital baseline parameter
    #     h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
    #     h_params.set_param("n_covid_general_wards", 20000)
    #     h_params.set_param("n_covid_icu_wards", 20000)
    #     h_params.write_params(SCENARIO_HOSPITAL_FILE)

    #     # Construct the compilation command and compile
    #     compile_command = "make clean; make all; make swig-all;"
    #     completed_compilation = subprocess.run([compile_command],
    #         shell = True,
    #         cwd = SRC_DIR,
    #         capture_output = True
    #         )

    #     # Construct the executable command
    #     EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} "+\
    #         f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

    #     # Call the model pipe output to file, read output file
    #     file_output = open(TEST_OUTPUT_FILE, "w")
    #     completed_run = subprocess.run([EXE], stdout = file_output, shell = True)
    #     df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

    #     # Check that the simulation ran
    #     assert len(df_output) != 0

    #     time_step_df = pd.read_csv(TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

    #     waiting_df = time_step_df['hospital_state'] == constant.WAITING
    #     waiting_df = time_step_df[waiting_df]

    #     assert len(waiting_df.index) == 0

    def test_all_hcw(self):
        """
        Set number of hcw to zero,
        assert there are no patient doctor interactions
        """

        population_size = 20000
        n_covid_general_wards = 20
        n_covid_icu_wards = 10

        # Adjust baseline parameter
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", population_size)
        params.write_params(SCENARIO_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_covid_general_wards", n_covid_general_wards)
        h_params.set_param("n_covid_icu_wards", n_covid_icu_wards)
        h_params.set_param("n_doctors_covid_general_ward", (population_size/4)/n_covid_general_wards)
        h_params.set_param("n_nurses_covid_general_ward", (population_size/4)/n_covid_general_wards)
        h_params.set_param("n_doctors_covid_icu_ward", (population_size/4)/n_covid_icu_wards)
        h_params.set_param("n_nurses_covid_icu_ward", (population_size/4)/n_covid_icu_wards)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)

        # Construct the compilation command and compile
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command],
            shell = True,
            cwd = SRC_DIR,
            capture_output = True
            )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {TEST_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        df_interactions = pd.read_csv(TEST_INTERACTIONS_FILE)

        df_doctor_patient_general_interactions = df_interactions[df_interactions["type"] == constant.HOSPITAL_DOCTOR_PATIENT_GENERAL]
        df_nurse_patient_general_interactions  = df_interactions[df_interactions["type"] == constant.HOSPITAL_NURSE_PATIENT_GENERAL]
        df_doctor_patient_icu_interactions = df_interactions[df_interactions["type"] == constant.HOSPITAL_DOCTOR_PATIENT_ICU]
        df_nurse_patient_icu_interactions  = df_interactions[df_interactions["type"] == constant.HOSPITAL_NURSE_PATIENT_ICU]

        df_doctor_patient_general_interactions = df_interactions[df_doctor_patient_general_interactions]
        df_nurse_patient_general_interactions = df_interactions[df_nurse_patient_general_interactions]
        df_doctor_patient_icu_interactions = df_interactions[df_doctor_patient_icu_interactions]
        df_nurse_patient_icu_interactions = df_interactions[df_nurse_patient_icu_interactions]

        assert len(df_doctor_patient_general_interactions.index) > 0
        assert len(df_nurse_patient_general_interactions.index) > 0
        assert len(df_doctor_patient_icu_interactions.index) > 0
        assert len(df_nurse_patient_icu_interactions.index) > 0

# Dylan update for file change
    def test_only_hcw_infections(self):
        """
        Only let infections occur in the hospital. Assert total new infections = total hospital infections.
        """

        # Adjust baseline parameter
        params = ParameterSet(TEST_DATA_FILE, line_number=1)
        params.set_param("infectious_rate", 0.0001)
        params.set_param("n_total", 20000)
        params.write_params(SCENARIO_FILE)

        # Construct the executable command
        EXE = f"{EXECUTABLE} {TEST_DATA_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {SCENARIO_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("general_infectivity_modifier", 57500.0)
        h_params.write_params(SCENARIO_HOSPITAL_FILE)

        # Construct the compilation command and compile
        compile_command = "make clean; make all; make swig-all;"
        completed_compilation = subprocess.run([compile_command],
            shell = True,
            cwd = SRC_DIR,
            capture_output = True
            )

        # Construct the executable command
        EXE = f"{EXECUTABLE} {SCENARIO_FILE} {PARAM_LINE_NUMBER} "+\
            f"{DATA_DIR_TEST} {TEST_HOUSEHOLD_FILE} {TEST_HOSPITAL_FILE}"

        # Call the model pipe output to file, read output file
        file_output = open(TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([EXE], stdout = file_output, shell = True)
        df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")

        # Check that the simulation ran
        assert len(df_output) != 0

        time_step_df = pd.read_csv(TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

        # Only keep instances where people have been infected after the zero time step
        never_infected_condition = time_step_df["disease_state"] != constant.EVENT_TYPES.UNINFECTED.value
        seed_infected_condition = time_step_df["time_infected"] != 0
        infected_df = time_step_df[never_infected_condition & seed_infected_condition]

        # Only keep instances where people that have been infected are hcw
        doctor_condition = infected_df["doctor_type"] == 1
        nurse_condition = infected_df["nurse_type"] == 1
        hcw_condition = infected_df[doctor_condition | nurse_condition]

        assert len(hcw_condition.index) == len(infected_df.index)