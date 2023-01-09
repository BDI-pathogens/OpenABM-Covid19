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


class TestClass(object):
    """
    Test class for checking that the model behaves as expected at parameter boundaries.
    """

    def test_zero_beds(self,tmp_path):
        """
        Set hospital beds to zero
        """

        # Adjust baseline parameter
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(tmp_path/constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_beds_covid_general_ward", 0)
        h_params.set_param("n_beds_covid_icu_ward", 0)
        h_params.write_params(tmp_path/constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        df_time_step = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

        n_patient_general = df_time_step["hospital_state"] == constant.EVENT_TYPES.GENERAL.value
        n_patient_general = df_time_step[n_patient_general]
        n_patient_general = len(n_patient_general.index)

        assert n_patient_general == 0

        n_patient_icu = df_time_step["hospital_state"] == constant.EVENT_TYPES.ICU.value
        n_patient_icu = df_time_step[n_patient_icu]
        n_patient_icu = len(n_patient_icu.index)

        assert n_patient_icu == 0

    def test_zero_general_wards(self,tmp_path):
        """
        Set hospital general wards to zero.
        Assert that there are no general patients.
        """

        # Adjust baseline parameter
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(tmp_path/constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_covid_general_wards", 0)
        h_params.set_param("hcw_mean_work_interactions", 0)
        h_params.write_params(tmp_path/constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        df_time_step = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

        n_patient_general = df_time_step["hospital_state"] == constant.EVENT_TYPES.GENERAL.value
        n_patient_general = df_time_step[n_patient_general]
        n_patient_general = len(n_patient_general.index)

        assert n_patient_general == 0


    def test_zero_icu_wards(self,tmp_path):
        """
        Set hospital icu wards to zero.
        Assert that there are no icu patients.
        """

        # Adjust baseline parameter
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(tmp_path/constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_covid_icu_wards", 0)
        h_params.set_param("hcw_mean_work_interactions", 0)
        h_params.write_params(tmp_path/constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        df_time_step = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

        n_patient_icu = df_time_step["hospital_state"] == constant.EVENT_TYPES.ICU.value
        n_patient_icu = df_time_step[n_patient_icu]
        n_patient_icu = len(n_patient_icu.index)

        assert n_patient_icu == 0


    def test_zero_hcw_patient_interactions(self,tmp_path):
        """
        Set patient required hcw interactions to zero.
        Assert that there are no interactions between hcw and patients
        """

        # Adjust baseline parameter
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.set_param("days_of_interactions", 10)
        params.set_param("end_time", 20)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(tmp_path/constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_patient_doctor_required_interactions_covid_general", 0)
        h_params.set_param("n_patient_nurse_required_interactions_covid_general_ward", 0)
        h_params.set_param("n_patient_doctor_required_interactions_covid_icu_ward", 0)
        h_params.set_param("n_patient_nurse_required_interactions_covid_icu_ward", 0)
        h_params.write_params(tmp_path/constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        df_interactions = pd.read_csv(tmp_path/constant.TEST_INTERACTION_FILE, comment="#", sep=",", skipinitialspace=True)

        df_doctor_patient_general_interactions = df_interactions[df_interactions["type"] == constant.HOSPITAL_DOCTOR_PATIENT_GENERAL]
        df_nurse_patient_general_interactions  = df_interactions[df_interactions["type"] == constant.HOSPITAL_NURSE_PATIENT_GENERAL]
        df_doctor_patient_icu_interactions     = df_interactions[df_interactions["type"] == constant.HOSPITAL_DOCTOR_PATIENT_ICU]
        df_nurse_patient_icu_interactions      = df_interactions[df_interactions["type"] == constant.HOSPITAL_NURSE_PATIENT_ICU]

        assert len(df_doctor_patient_general_interactions) == 0
        assert len(df_nurse_patient_general_interactions) == 0
        assert len(df_doctor_patient_icu_interactions) == 0
        assert len(df_nurse_patient_icu_interactions) == 0

    def test_zero_max_hcw_daily_interactions(self,tmp_path):
        """
        Set healthcare workers max daily interactions (with patients) to 0.
        Assert that there are no disease transition from patient to healthcare workers
        or any interactions.
        """

        # Adjust baseline parameter
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.set_param("days_of_interactions", 10)
        params.set_param("end_time", 20)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(tmp_path/constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("max_hcw_daily_interactions", 0)
        h_params.write_params(tmp_path/constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        df_transmissions_output = pd.read_csv(tmp_path/constant.TEST_TRANSMISSION_FILE)

        # Get healthcare workers
        healthcare_workers = df_transmissions_output["worker_type_recipient"] != constant.NOT_HEALTHCARE_WORKER
        healthcare_workers = df_transmissions_output[healthcare_workers]

        # Check that no healthcare workers have been infected by a patient
        for index, row in healthcare_workers.iterrows():
            assert row["hospital_state_source"] not in [constant.EVENT_TYPES.GENERAL.value, constant.EVENT_TYPES.ICU.value]

        df_interactions = pd.read_csv(tmp_path/constant.TEST_INTERACTION_FILE,comment="#", sep=",", skipinitialspace=True)

        df_doctor_patient_general_interactions = df_interactions[df_interactions["type"] == constant.HOSPITAL_DOCTOR_PATIENT_GENERAL]
        df_nurse_patient_general_interactions = df_interactions[df_interactions["type"] == constant.HOSPITAL_NURSE_PATIENT_GENERAL]
        df_doctor_patient_icu_interactions = df_interactions[df_interactions["type"] == constant.HOSPITAL_DOCTOR_PATIENT_ICU]
        df_nurse_patient_icu_interactions = df_interactions[df_interactions["type"] == constant.HOSPITAL_NURSE_PATIENT_ICU]

        assert len(df_doctor_patient_general_interactions) == 0
        assert len(df_nurse_patient_general_interactions) == 0
        assert len(df_doctor_patient_icu_interactions) == 0
        assert len(df_nurse_patient_icu_interactions) == 0


    def test_zero_hospitalised_waiting_mod(self,tmp_path):
        """
        Set hospitalised_waiting_mod to zero.
        Assert that everybody in waiting state is waiting or recovered in
        the next time steps.
        """

        # Adjust baseline parameter
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(tmp_path/constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("hospitalised_waiting_mod", 0.0)
        h_params.set_param("n_beds_covid_general_ward", 0)
        h_params.set_param("n_beds_covid_icu_ward", 0)
        h_params.write_params(tmp_path/constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment="#", sep=",")


        time_step_df = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

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


    def test_zero_critical_waiting_mod(self,tmp_path):
        """
        Set critical_waiting_mod to zero.
        Assert that everybody in waiting state is waiting or recovered in
        the next time steps.
        """

        # Adjust baseline parameter
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("location_death_icu_20_29", 1)
        params.set_param("location_death_icu_30_39", 1)
        params.set_param("location_death_icu_40_49", 1)
        params.set_param("location_death_icu_50_59", 1)
        params.set_param("location_death_icu_60_69", 1)
        params.set_param("location_death_icu_70_79", 1)
        params.set_param("location_death_icu_80", 1)
        params.set_param("hospital_on", 1)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(tmp_path/constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("critical_waiting_mod", 0.0)
        h_params.set_param("n_beds_covid_general_ward", 0)
        h_params.set_param("n_beds_covid_icu_ward", 0)
        h_params.write_params(tmp_path/constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        time_step_df = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

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


    def test_100_waiting_mods(self,tmp_path):
        """
        Set hospitalised_waiting_mod and critical_waiting to zero.
        Assert that everybody in waiting state is in waiting, critical or mortuary in the next time steps
        """

        # Adjust baseline parameter
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(tmp_path/constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("hospitalised_waiting_mod", 100.0)
        h_params.set_param("critical_waiting_mod", 100.0)
        h_params.set_param("n_beds_covid_general_ward", 0)
        h_params.set_param("n_beds_covid_icu_ward", 0)
        h_params.write_params(tmp_path/constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        time_step_df = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

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


    def test_no_space_limit_beds(self,tmp_path):
        """
        Set number of beds in each ward to the population size.
        Assert that nobody enters the waiting state.
        """

        # Adjust baseline parameter
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 20000)
        params.set_param("hospital_on", 1)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(tmp_path/constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_beds_covid_general_ward", 20000)
        h_params.set_param("n_beds_covid_icu_ward", 20000)
        h_params.write_params(tmp_path/constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        time_step_df = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)

        waiting_df = time_step_df["hospital_state"] == constant.EVENT_TYPES.WAITING.value
        waiting_df = time_step_df[waiting_df]

        assert len(waiting_df.index) == 0

    def test_many_hcw(self,tmp_path):
        """
        Set a third of the population in the simulation to be healthcare workers.
        Assert the model still runs and there are healthcareworker - patient interactions.
        """

        hcw_population_size = 10000
        n_covid_general_wards = 20
        n_covid_icu_wards = 10


        # Adjust baseline parameter
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number=1)
        params.set_param("n_total", 30000)
        params.set_param("hospital_on", 1)
        params.set_param("days_of_interactions", 10)
        params.set_param("end_time", 50)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Adjust hospital baseline parameter
        h_params = ParameterSet(tmp_path/constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("n_covid_general_wards", n_covid_general_wards)
        h_params.set_param("n_covid_icu_wards", n_covid_icu_wards)
        h_params.set_param("n_doctors_covid_general_ward", int((hcw_population_size/4)/n_covid_general_wards))
        h_params.set_param("n_nurses_covid_general_ward", int((hcw_population_size/4)/n_covid_general_wards))
        h_params.set_param("n_doctors_covid_icu_ward", int((hcw_population_size/4)/n_covid_icu_wards))
        h_params.set_param("n_nurses_covid_icu_ward", int((hcw_population_size/4)/n_covid_icu_wards))
        h_params.write_params(tmp_path/constant.TEST_HOSPITAL_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        df_interactions = pd.read_csv(tmp_path/constant.TEST_INTERACTION_FILE)

        interaction_type_HOSPITAL_DOCTOR_PATIENT_GENERAL = constant.HOSPITAL_DOCTOR_PATIENT_GENERAL
        interaction_type_HOSPITAL_NURSE_PATIENT_GENERAL = constant.HOSPITAL_NURSE_PATIENT_GENERAL
        interaction_type_HOSPITAL_DOCTOR_PATIENT_ICU = constant.HOSPITAL_DOCTOR_PATIENT_ICU
        interaction_type_HOSPITAL_NURSE_PATIENT_ICU = constant.HOSPITAL_NURSE_PATIENT_ICU

        df_doctor_patient_general_interactions = df_interactions.query("type == @interaction_type_HOSPITAL_DOCTOR_PATIENT_GENERAL")
        df_nurse_patient_general_interactions = df_interactions.query("type == @interaction_type_HOSPITAL_NURSE_PATIENT_GENERAL")
        df_doctor_patient_icu_interactions = df_interactions.query("type == @interaction_type_HOSPITAL_DOCTOR_PATIENT_ICU")
        df_nurse_patient_icu_interactions = df_interactions.query("type == @interaction_type_HOSPITAL_NURSE_PATIENT_ICU")

        assert len(df_doctor_patient_general_interactions.index) > 0
        assert len(df_nurse_patient_general_interactions.index) > 0
        assert len(df_doctor_patient_icu_interactions.index) > 0
        assert len(df_nurse_patient_icu_interactions.index) > 0

    def test_transmission_doctor_general(self,tmp_path):
        """
        When general doctor-patient transmission is very high and no other forms of tranmission can occur for doctors,
        check that all doctors become infected when the general ward is overloaded with patients.
        """

        # Set general doctor-patient infectivity to be really high
        h_params = ParameterSet(tmp_path/constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("relative_transmission_doctor_patient_general", 100.0)

        # Set other transmission types in hospitals to zero.
        h_params.set_param("relative_transmission_hospital_work", 0.0)
        h_params.set_param("relative_transmission_nurse_patient_general", 0.0)
        h_params.set_param("relative_transmission_doctor_patient_icu", 0.0)
        h_params.set_param("relative_transmission_nurse_patient_icu", 0.0)
        h_params.set_param("n_hospitals", 1)
        h_params.set_param("n_covid_general_wards", 5)
        h_params.set_param("n_doctors_covid_general_ward", 5)
        h_params.write_params(tmp_path/constant.TEST_HOSPITAL_FILE)

        # Set transmission types elsewhere that doctors are associated with to zero
        # Also set the number of infections to be really high.
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number=1)
        params.set_param("relative_transmission_household", 0.0)
        params.set_param("relative_transmission_random", 0.0)
        params.set_param("n_total", 50000)
        params.set_param("n_seed_infection", 45000)
        params.set_param("hospital_on", 1)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        # Get all general doctor
        df_time_step = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)
        df_all_hcw = df_time_step.query("doctor_type == 1 & ward_type == 0")
        list_all_hcw = df_all_hcw.pdx.unique()

        for hcw in range(len(list_all_hcw)):

            hcw_pdx = list_all_hcw[hcw]
            hcw_all_time_steps = df_time_step.query("pdx == @hcw_pdx")
            disease_state = hcw_all_time_steps["disease_state"].max()

            assert disease_state > 0

    def test_transmission_nurse_general(self,tmp_path):
        """
        Set general nurse-patient transmission high and switch off other forms of tranmission for doctors.
        Assert that all nurses become infected when the general ward is overloaded with patients.
        """

        # Set general nurse-patient infectivity to be really high.
        h_params = ParameterSet(tmp_path/constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("relative_transmission_nurse_patient_general", 100.0)

        # Set other transmission types in hospitals to zero.
        h_params.set_param("relative_transmission_hospital_work", 0.0)
        h_params.set_param("relative_transmission_doctor_patient_general", 0.0)
        h_params.set_param("relative_transmission_doctor_patient_icu", 0.0)
        h_params.set_param("relative_transmission_nurse_patient_icu", 0.0)
        h_params.set_param("n_hospitals", 1)
        h_params.set_param("n_covid_general_wards", 5)
        h_params.set_param("n_nurses_covid_general_ward", 5)
        h_params.write_params(tmp_path/constant.TEST_HOSPITAL_FILE)

        # Set transmission types elsewhere that nurses are associated with to zero.
        # Also set the number of infections to be really high.
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number=1)
        params.set_param("relative_transmission_household", 0.0)
        params.set_param("relative_transmission_random", 0.0)
        params.set_param("n_total", 50000)
        params.set_param("n_seed_infection", 45000)
        params.set_param("hospital_on", 1)

        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        # Get all general nurses
        df_time_step = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)
        df_all_hcw = df_time_step.query("nurse_type == 1 & ward_type == 0")
        list_all_hcw = df_all_hcw.pdx.unique()

        for hcw in range(len(list_all_hcw)):

            hcw_pdx = list_all_hcw[hcw]
            hcw_all_time_steps = df_time_step.query("pdx == @hcw_pdx")
            disease_state = hcw_all_time_steps["disease_state"].max()

            assert disease_state > 0

    def test_transmission_doctor_icu(self,tmp_path):
        """
        Set icu doctor-patient transmission is high and switch off other forms of transmission for doctors.
        Assert that all doctors become infected when the icu ward is overloaded with patients.
        """

        # Set icu doctor-patient infectivity to be high
        h_params = ParameterSet(tmp_path/constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("relative_transmission_doctor_patient_icu", 100.0)

        # Set other transmission types in hospitals to zero.
        h_params.set_param("relative_transmission_hospital_work", 0.0)
        h_params.set_param("relative_transmission_doctor_patient_general", 0.0)
        h_params.set_param("relative_transmission_nurse_patient_general", 0.0)
        h_params.set_param("relative_transmission_nurse_patient_icu", 0.0)
        h_params.set_param("n_hospitals", 1)
        h_params.set_param("n_covid_icu_wards", 10)
        h_params.set_param("n_covid_general_wards", 1)
        h_params.set_param("n_doctors_covid_icu_ward", 3)
        h_params.write_params(tmp_path/constant.TEST_HOSPITAL_FILE)

        # Set transmission types elsewhere that doctors are associated with to zero.
        # Also set the number of infections to be really high.
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number=1)
        params.set_param("relative_transmission_household", 0.0)
        params.set_param("relative_transmission_random", 0.0)
        params.set_param("n_total", 50000)
        params.set_param("n_seed_infection", 45000)
        params.set_param("hospital_on", 1)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        # Get all icu doctors
        df_time_step = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)
        df_all_hcw = df_time_step.query("doctor_type == 1 & ward_type == 1")
        list_all_hcw = df_all_hcw.pdx.unique()

        for hcw in range(len(list_all_hcw)):

            hcw_pdx = list_all_hcw[hcw]
            hcw_all_time_steps = df_time_step.query("pdx == @hcw_pdx")
            disease_state = hcw_all_time_steps["disease_state"].max()

            assert disease_state > 0


    def test_transmission_nurse_icu(self,tmp_path):
        """
        Set icu nurse-patient transmission very high and switch off other forms of transmission for nurses.
        Assert that all nurses become infected when the icu ward is overloaded with patients.
        """

        # Set icu nurse-patient infectivity to be high.
        h_params = ParameterSet(tmp_path/constant.TEST_HOSPITAL_FILE, line_number=1)
        h_params.set_param("relative_transmission_nurse_patient_icu", 100.0)

        # Set other transmission types in hospitals to zero.
        h_params.set_param("relative_transmission_hospital_work", 0.0)
        h_params.set_param("relative_transmission_doctor_patient_general", 0.0)
        h_params.set_param("relative_transmission_nurse_patient_general", 0.0)
        h_params.set_param("relative_transmission_doctor_patient_icu", 0.0)
        h_params.set_param("relative_transmission_doctor_patient_icu", 0.0)
        h_params.set_param("n_hospitals", 1)
        h_params.set_param("n_covid_icu_wards", 5)
        h_params.set_param("n_covid_general_wards", 1)
        h_params.set_param("n_nurses_covid_icu_ward", 5)
        h_params.write_params(tmp_path/constant.TEST_HOSPITAL_FILE)

        # Set transmission types elsewhere that doctors are associated with to zero.
        # Also set the number of infections to be high.
        params = ParameterSet(tmp_path/constant.TEST_DATA_FILE, line_number=1)
        params.set_param("relative_transmission_household", 0.0)
        params.set_param("relative_transmission_random", 0.0)
        params.set_param("n_total", 50000)
        params.set_param("n_seed_infection", 45000)
        params.set_param("hospital_on", 1)
        params.write_params(tmp_path/constant.TEST_DATA_FILE)

        # Call the model pipe output to file, read output file
        file_output = open(tmp_path/constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command_tmp(tmp_path)], stdout = file_output, shell = True)
        df_output = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE, comment="#", sep=",")

        # Get all icu nurses
        df_time_step = pd.read_csv(tmp_path/constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)
        df_all_hcw = df_time_step.query("nurse_type == 1 & ward_type == 1")
        list_all_hcw = df_all_hcw.pdx.unique()

        for hcw in range(len(list_all_hcw)):

            hcw_pdx = list_all_hcw[hcw]
            hcw_all_time_steps = df_time_step.query("pdx == @hcw_pdx")
            disease_state = hcw_all_time_steps["disease_state"].max()

            assert disease_state > 0
