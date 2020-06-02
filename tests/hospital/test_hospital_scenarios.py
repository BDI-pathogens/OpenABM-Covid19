#!/usr/bin/env python3
"""
Tests of the individual-based model, COVID19-IBM

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be 
run by calling 'pytest' from project folder.  

Created: April 2020
Author: Dylan Feldner-Busztin
"""

from parameters import ParameterSet
from tests import constant
import subprocess, pytest, os, sys
import numpy as np, pandas as pd

# TEST_DIR = os.path.dirname(os.path.realpath(__file__))
# TEST_DIR = TEST_DIR.replace("hospital", "")
# TEST_HOSPITAL_FILE = TEST_DIR + "data/hospital_baseline_parameters.csv"
# TEST_DATA_FILE = TEST_DIR + "data/baseline_parameters.csv"
# PARAM_LINE_NUMBER = 1
# DATA_DIR_TEST = TEST_DIR + "data"
# TEST_HOUSEHOLD_FILE = TEST_DIR + "data/baseline_household_demographics.csv"
# TEST_OUTPUT_FILE = TEST_DIR + "data/test_output.csv"
# TEST_OUTPUT_FILE_HOSPITAL = TEST_DIR + "data/test_hospital_output.csv"
# TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP = TEST_DIR + "data/time_step_hospital_output.csv"
# TEST_INTERACTIONS_FILE = TEST_DIR + "data/interactions_Run1.csv"
# TEST_INDIVIDUAL_FILE = TEST_DIR + "data/individual_file_Run1.csv"
# TEST_HCW_FILE = TEST_DIR + "data/ward_output.csv"
# TEST_TRANSMISSION_FILE = TEST_DIR + "data/transmission_Run1.csv"
# SRC_DIR = TEST_DIR.replace("tests", "") + "src"
# EXECUTABLE = SRC_DIR + "/covid19ibm.exe"

# # Files with adjusted parameters for each scenario
# SCENARIO_FILE = TEST_DIR + "/data/scenario_baseline_parameters.csv"
# SCENARIO_HOSPITAL_FILE = TEST_DIR + "/data/scenario_hospital_baseline_parameters.csv"

# # Use parameter file from Python C interface to adjust parameters
# PYTHON_C_DIR = TEST_DIR.replace("tests","") + "src/COVID19"
# sys.path.append(PYTHON_C_DIR)

class TestClass(object):
    """
    Test class for checking
    """

    def test_zero_beds(self):
        """
        Set hospital beds to zero
        """

        # Adjust baseline parameter
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_beds_covid_general_ward", 0)
        h_params.set_param("n_beds_covid_icu_ward", 0)
        h_params.write_params(constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        df_time_step = pd.read_csv(constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

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
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_covid_general_wards", 0)
        h_params.write_params(constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        df_time_step = pd.read_csv(constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

        n_patient_general = df_time_step["hospital_state"] == constant.EVENT_TYPES.GENERAL.value
        n_patient_general = df_time_step[n_patient_general]
        n_patient_general = len(n_patient_general.index)

        assert n_patient_general == 0


    def test_zero_icu_wards(self):
        """
        Set hospital icu wards to zero, check there are no icu patients
        """

        # Adjust baseline parameter
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_covid_icu_wards", 0)
        h_params.write_params(constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        df_time_step = pd.read_csv(constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

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

        # Adjust baseline parameter
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_patient_doctor_required_interactions_covid_general", 0)
        h_params.set_param("n_patient_nurse_required_interactions_covid_general_ward", 0)
        h_params.set_param("n_patient_doctor_required_interactions_covid_icu_ward", 0)
        h_params.set_param("n_patient_nurse_required_interactions_covid_icu_ward", 0)
        h_params.write_params(constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        df_interactions = pd.read_csv(constant.TEST_INTERACTION_FILE,
                                      comment="#", sep=",", skipinitialspace=True)

        df_doctor_patient_general_interactions = df_interactions[df_interactions["type"] == constant.HOSPITAL_DOCTOR_PATIENT_GENERAL]
        df_nurse_patient_general_interactions  = df_interactions[df_interactions["type"] == constant.HOSPITAL_NURSE_PATIENT_GENERAL]
        df_doctor_patient_icu_interactions     = df_interactions[df_interactions["type"] == constant.HOSPITAL_DOCTOR_PATIENT_ICU]
        df_nurse_patient_icu_interactions      = df_interactions[df_interactions["type"] == constant.HOSPITAL_NURSE_PATIENT_ICU]

        assert len(df_doctor_patient_general_interactions) == 0
        assert len(df_nurse_patient_general_interactions) == 0
        assert len(df_doctor_patient_icu_interactions) == 0
        assert len(df_nurse_patient_icu_interactions) == 0


    def test_hospital_infectivity_modifiers_zero(self):
        """
        Set patient infectivity modifiers to zero and test that no healthcare
        workers are infected by a patient
        """

        # Adjust baseline parameter
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("waiting_infectivity_modifier", 0)
        h_params.set_param("general_infectivity_modifier", 0)
        h_params.set_param("icu_infectivity_modifier", 0)
        h_params.write_params(constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        df_transmissions_output = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        healthcare_workers = df_transmissions_output["worker_type_recipient"] != constant.NOT_HEALTHCARE_WORKER
        healthcare_workers = df_transmissions_output[healthcare_workers]

        # check that no healthcare workers have been infected by a patient
        for index, row in healthcare_workers.iterrows():
            assert row["hospital_state_source"] not in [constant.EVENT_TYPES.GENERAL.value, constant.EVENT_TYPES.ICU.value]

    def test_hospital_infectivity_modifiers_max(self):
        """
        Set patient infectivity modifiers to 100 and test that all healthcare
        workers who have interacted with a patient get infected
        """

        # Adjust baseline parameter
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("waiting_infectivity_modifier", 100)
        h_params.set_param("general_infectivity_modifier", 100)
        h_params.set_param("icu_infectivity_modifier", 100)
        h_params.write_params(constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        df_individual_output = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_int = pd.read_csv(constant.TEST_INTERACTION_FILE, comment="#", sep=",", skipinitialspace=True)
        time_step_df = pd.read_csv(constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

        #get all healthcare workers who have interacted with patients
        hcw_with_patient_interaction = (df_int["worker_type_1"] != constant.NOT_HEALTHCARE_WORKER) & (df_int["type"] > constant.HOSPITAL_WORK) & (df_int["type"] <= constant.HOSPITAL_NURSE_PATIENT_ICU)
        hcw_with_patient_interaction = df_int[hcw_with_patient_interaction]

        # make sure these healthcare workers are infected at some point
        for index, row in hcw_with_patient_interaction.iterrows():
            hcw_infected = (time_step_df["pdx"] == row["ID_1"]) & (time_step_df["disease_state"] >= constant.EVENT_TYPES.PRESYMPTOMATIC.value)
            hcw_infected = time_step_df[hcw_infected]
            assert len(hcw_infected.index) > 0

    def test_max_hcw_daily_interactions(self):
        """
        Set healthcare workers max daily interactions (with patients) to 0 and
        check that there are no disease transition from patient to healthcare workers
        or any interactions
        """

        # Adjust baseline parameter
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("max_hcw_daily_interactions", 0)
        h_params.write_params(constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        df_transmissions_output = pd.read_csv(constant.TEST_TRANSMISSION_FILE)

        # get healthcare workers
        healthcare_workers = df_transmissions_output["worker_type_recipient"] != constant.NOT_HEALTHCARE_WORKER
        healthcare_workers = df_transmissions_output[healthcare_workers]

        # check that no healthcare workers have been infected by a patient
        for index, row in healthcare_workers.iterrows():
            assert row["hospital_state_source"] not in [constant.EVENT_TYPES.GENERAL.value, constant.EVENT_TYPES.ICU.value]

        df_interactions = pd.read_csv(constant.TEST_INTERACTION_FILE,
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
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("hospitalised_waiting_mod", 0.0)
        h_params.set_param("n_beds_covid_general_ward", 0)
        h_params.set_param("n_beds_covid_icu_ward", 0)
        h_params.write_params(constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment="#", sep=",")


        time_step_df = pd.read_csv(constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

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
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("location_death_icu_20_29", 1)
        params.set_param("location_death_icu_30_39", 1)
        params.set_param("location_death_icu_40_49", 1)
        params.set_param("location_death_icu_50_59", 1)
        params.set_param("location_death_icu_60_69", 1)
        params.set_param("location_death_icu_70_79", 1)
        params.set_param("location_death_icu_80", 1)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("critical_waiting_mod", 0.0)
        h_params.set_param("n_beds_covid_general_ward", 0)
        h_params.set_param("n_beds_covid_icu_ward", 0)
        h_params.write_params(constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        time_step_df = pd.read_csv(constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

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
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("hospitalised_waiting_mod", 100.0)
        h_params.set_param("critical_waiting_mod", 100.0)
        h_params.set_param("n_beds_covid_general_ward", 0)
        h_params.set_param("n_beds_covid_icu_ward", 0)
        h_params.write_params(constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        time_step_df = pd.read_csv(constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

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
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_beds_covid_general_ward", 20000)
        h_params.set_param("n_beds_covid_icu_ward", 20000)
        h_params.write_params(constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        time_step_df = pd.read_csv(constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

        waiting_df = time_step_df["hospital_state"] == constant.EVENT_TYPES.WAITING.value
        waiting_df = time_step_df[waiting_df]

        assert len(waiting_df.index) == 0

    def test_max_hcw(self):
        """
        half population in the simulation is a hcw. Assert no hcw - patient interactions.
        """

        hcw_population_size = 5000
        n_covid_general_wards = 20
        n_covid_icu_wards = 10

        # Adjust baseline parameter
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 10000)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_covid_general_wards", n_covid_general_wards)
        h_params.set_param("n_covid_icu_wards", n_covid_icu_wards)
        h_params.set_param("n_doctors_covid_general_ward", int((hcw_population_size/4)/n_covid_general_wards))
        h_params.set_param("n_nurses_covid_general_ward", int((hcw_population_size/4)/n_covid_general_wards))
        h_params.set_param("n_doctors_covid_icu_ward", int((hcw_population_size/4)/n_covid_icu_wards))
        h_params.set_param("n_nurses_covid_icu_ward", int((hcw_population_size/4)/n_covid_icu_wards))
        h_params.write_params(constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        df_interactions = pd.read_csv(constant.TEST_INTERACTION_FILE)

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

    def test_transmission_doctor_general(self):
        """
        When general doctor-patient transmission is very high and no other forms of tranmission can occur for doctors,
        check that all doctors become infected when the general ward is overloaded with patients.
        """

        # Set general doctor-patient infectivity to be really high
        h_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("relative_transmission_doctor_patient_general", 100.0)

        # Set other transmission types in hospitals to zero.
        h_params.set_param("relative_transmission_hospital_work", 0.0)
        h_params.set_param("relative_transmission_nurse_patient_general", 0.0)
        h_params.set_param("relative_transmission_doctor_patient_icu", 0.0)
        h_params.set_param("relative_transmission_nurse_patient_icu", 0.0)
        h_params.write_params(constant.TEST_HOSPITAL_FILE)

        # Set transmission types elsewhere that doctors are associated with to zero
        # Also set the number of infections to be really high.
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("relative_transmission_household", 0.0)
        params.set_param("relative_transmission_random", 0.0)
        params.set_param("n_seed_infection", 7500)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        # Get all uninfected doctors working in the general ward.
        df_individual_output = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_transmission_output = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_combined_output = pd.merge(df_individual_output, df_transmission_output,
                                      left_on = "ID", right_on = "ID_recipient", how = "left")
        n_doctors = df_combined_output["worker_type"] == 0
        time_infected = df_combined_output["time_infected"] == -1
        n_general = df_combined_output["assigned_worker_ward_type"] == 0
        n_general_doctors = df_combined_output[n_doctors & n_general & time_infected]

        # Check that all doctors assigned to the general ward end up being infected.
        assert(len(n_general_doctors.index) == 0)

    def test_transmission_nurse_general(self):
        """
        When general nurse-patient transmission is very high and no other forms of tranmission can occur for doctors,
        check that all nurses become infected when the general ward is overloaded with patients.
        """

        # Set general nurse-patient infectivity to be really high.
        h_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("relative_transmission_nurse_patient_general", 100.0)

        # Set other transmission types in hospitals to zero.
        h_params.set_param("relative_transmission_hospital_work", 0.0)
        h_params.set_param("relative_transmission_doctor_patient_general", 0.0)
        h_params.set_param("relative_transmission_doctor_patient_icu", 0.0)
        h_params.set_param("relative_transmission_nurse_patient_icu", 0.0)
        h_params.write_params(constant.TEST_HOSPITAL_FILE)

        # Set transmission types elsewhere that nurses are associated with to zero.
        # Also set the number of infections to be really high.
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("relative_transmission_household", 0.0)
        params.set_param("relative_transmission_random", 0.0)
        params.set_param("n_seed_infection", 7500)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        # Get all uninfected doctors working in the general ward.
        df_individual_output = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_transmission_output = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_combined_output = pd.merge(df_individual_output, df_transmission_output,
                                      left_on = "ID", right_on = "ID_recipient", how = "left")
        n_nurses = df_combined_output["worker_type"] == 1
        time_infected = df_combined_output["time_infected"] == -1
        n_general = df_combined_output["assigned_worker_ward_type"] == 0
        n_general_nurses = df_combined_output[n_nurses & n_general & time_infected]

        #Check that all nurses assigned to the general ward end up being infected.
        assert(len(n_general_nurses.index) == 0)

    def test_transmission_doctor_icu(self):
        """
        When icu doctor-patient transmission is very high and no other forms of tranmission can occur for doctors,
        check that all doctors become infected when the icu ward is overloaded with patients.
        """

        # Set icu doctor-patient infectivity to be really high
        h_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("relative_transmission_doctor_patient_icu", 100.0)

        # Set other transmission types in hospitals to zero.
        h_params.set_param("relative_transmission_hospital_work", 0.0)
        h_params.set_param("relative_transmission_doctor_patient_general", 0.0)
        h_params.set_param("relative_transmission_nurse_patient_general", 0.0)
        h_params.set_param("relative_transmission_nurse_patient_icu", 0.0)
        h_params.write_params(constant.TEST_HOSPITAL_FILE)

        # Set transmission types elsewhere that doctors are associated with to zero.
        # Also set the number of infections to be really high.
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("relative_transmission_household", 0.0)
        params.set_param("relative_transmission_random", 0.0)
        params.set_param("n_seed_infection", 7500)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        # Get all uninfected doctors working in the general ward.
        df_individual_output = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_transmission_output = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_combined_output = pd.merge(df_individual_output, df_transmission_output,
                                      left_on = "ID", right_on = "ID_recipient", how = "left")
        n_doctors = df_combined_output["worker_type"] == 0
        time_infected = df_combined_output["time_infected"] == -1
        n_icu = df_combined_output["assigned_worker_ward_type"] == 1
        n_icu_doctors = df_combined_output[n_doctors & n_icu & time_infected]

        #Check that all doctors assigned to the icu ward end up being infected.
        assert(len(n_icu_doctors.index) == 0)

    def test_transmission_nurse_icu(self):
        """
        When icu nurse-patient transmission is very high and no other forms of tranmission can occur for nurses,
        check that all nurses become infected when the icu ward is overloaded with patients.
        """

        # Set icu nurse-patient infectivity to be really high.
        h_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("relative_transmission_nurse_patient_icu", 100.0)

        # Set other transmission types in hospitals to zero.
        h_params.set_param("relative_transmission_hospital_work", 0.0)
        h_params.set_param("relative_transmission_doctor_patient_general", 0.0)
        h_params.set_param("relative_transmission_nurse_patient_general", 0.0)
        h_params.set_param("relative_transmission_doctor_patient_icu", 0.0)
        h_params.write_params(constant.TEST_HOSPITAL_FILE)

        # Set transmission types elsewhere that doctors are associated with to zero.
        # Also set the number of infections to be really high.
        params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
        params.set_param("relative_transmission_household", 0.0)
        params.set_param("relative_transmission_random", 0.0)
        params.set_param("n_seed_infection", 7500)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)
        df_output = pd.read_csv(constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        # Get all uninfected nurses working in the general ward.
        df_individual_output = pd.read_csv(constant.TEST_INDIVIDUAL_FILE)
        df_transmission_output = pd.read_csv(constant.TEST_TRANSMISSION_FILE)
        df_combined_output = pd.merge(df_individual_output, df_transmission_output,
                                      left_on = "ID", right_on = "ID_recipient", how = "left")
        n_nurses = df_combined_output["worker_type"] == 1
        time_infected = df_combined_output["time_infected"] == -1
        n_icu = df_combined_output["assigned_worker_ward_type"] == 1
        n_icu_nurses = df_combined_output[n_nurses & n_icu & time_infected]

        #Check that all nurses assigned to the general ward end up being infected.
        assert(len(n_icu_nurses.index) == 0)
