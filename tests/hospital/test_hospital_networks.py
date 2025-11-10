#!/usr/bin/env python3
"""
Tests of the individual-based model, COVID19-IBM, using the individual file
Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be
run by calling 'pytest' from project folder.
Created: June 2020
Author: Kelvin van Vuuren
"""

import pytest, sys, subprocess, shutil, os
import numpy as np, pandas as pd
from scipy import optimize

sys.path.append("src/COVID19")
from tests import utilities as utils
from parameters import ParameterSet

from tests import constant


def get_interaction_type(ward_type, hcw_type):
    if ward_type == constant.HOSPITAL_WARD_TYPES.COVID_GENERAL.value:
        if hcw_type == constant.HCW_TYPES.DOCTOR.value:
            return constant.HOSPITAL_DOCTOR_PATIENT_GENERAL
        if hcw_type == constant.HCW_TYPES.NURSE.value:
            return constant.HOSPITAL_NURSE_PATIENT_GENERAL
    if ward_type == constant.HOSPITAL_WARD_TYPES.COVID_ICU.value:
        if hcw_type == constant.HCW_TYPES.DOCTOR.value:
            return constant.HOSPITAL_DOCTOR_PATIENT_ICU
        if hcw_type == constant.HCW_TYPES.NURSE.value:
            return constant.HOSPITAL_NURSE_PATIENT_ICU

class TestClass(object):
    """
    Test class for checking
    """

    def test_hcw_patient_networks(self):
        """
        Test that the correct number of interactions between healthcare workers and patients have
        occurred each timestep across both general / icu wards
        """
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params.set_param("hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)
        end_time = int(params.get_param("end_time"))

        hospital_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number=1)

        patient_required_hcw_interaction = [
            [int(hospital_params.get_param("n_patient_doctor_required_interactions_covid_general")), int(hospital_params.get_param("n_patient_nurse_required_interactions_covid_general_ward"))],
            [int(hospital_params.get_param("n_patient_doctor_required_interactions_covid_icu_ward")), int(hospital_params.get_param("n_patient_nurse_required_interactions_covid_icu_ward"))]
        ]
        n_wards = [int(hospital_params.get_param("n_covid_general_wards")), int(hospital_params.get_param("n_covid_icu_wards"))]
        max_hcw_interactions = int(hospital_params.get_param("max_hcw_daily_interactions"))

        # Call the model pipe output to file, read output file
        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout=file_output, shell=True)

        df_time_step = pd.read_csv(constant.TEST_OUTPUT_FILE_HOSPITAL_TIME_STEP)
        df_interactions = pd.read_csv(constant.TEST_OUTPUT_FILE_HOSPITAL_INTERACTIONS)

        for time_step in range(end_time):
            next_time_step = time_step + 1
            for ward_type in constant.HOSPITAL_WARD_TYPES:
                ward_type = ward_type.value
                for ward_index in range(n_wards[ward_type]):
                    get_patients_query_str = "time_step == @time_step & patient_type == 1 & ward_type == @ward_type & ward_idx == @ward_index"
                    current_ward_patients = df_time_step.query(get_patients_query_str)
                    for hcw_type in constant.HCW_TYPES:
                        hcw_type = hcw_type.value
                        hcw_type_str = "doctor_type"
                        if hcw_type == constant.HCW_TYPES.NURSE.value:
                            hcw_type_str = "nurse_type"
                        hcw_working_query_str = "time_step == @time_step & " + hcw_type_str + " == 1 & ward_type == @ward_type & ward_idx == @ward_index & is_working == 1"
                        hcw_working = df_time_step.query(hcw_working_query_str)

                        interactions_per_hcw = 0.0
                        if len(current_ward_patients) > 0 and len(hcw_working) > 0:
                            interactions_per_hcw = (len(current_ward_patients) * patient_required_hcw_interaction[ward_type][hcw_type])/len(hcw_working)
                            if interactions_per_hcw > max_hcw_interactions:
                                interactions_per_hcw = max_hcw_interactions
                            network_interaction_type = get_interaction_type(ward_type, hcw_type)
                            n_expected_interactions = int(round(interactions_per_hcw * len(hcw_working)))
                            actual_interactions_query_str = "worker_type_1 == @hcw_type & time_step == @next_time_step & ward_type_1 == @ward_type & ward_idx_1 == @ward_index & interaction_type == @network_interaction_type"
                            actual_interactions = df_interactions.query(actual_interactions_query_str)
                            n_actual_interactions = len(actual_interactions)
                            assert n_expected_interactions == n_actual_interactions, 'expected interactions does not match actual interactions! timestep: ' + str(next_time_step) + ' for ward: type ' + str(ward_type) + ', index ' + str(ward_index) + ', worker type ' + str(hcw_type)


    def test_hcw_network( self ):
        """
        Test to check that healthcare work connections are on the correct network and
        that they have correct number on average. When counting connections we count each end
        separately.
        """

        # absolute tolerance
        tolerance = 0.035

        ageTypeMap1 = pd.DataFrame(
            data = { "age_group_1": constant.AGES, "age_type_1": constant.AGE_TYPES } );
        ageTypeMap2 = pd.DataFrame(
            data = { "age_group_2": constant.AGES, "age_type_2": constant.AGE_TYPES } );

        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        params = utils.turn_off_interventions(params,1)
        params.set_param("n_total", 40000)
        params.set_param( "hospital_on", 1)
        params.write_params(constant.TEST_DATA_FILE)

        hospital_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number = 1)
        hospital_params.set_param( "hcw_mean_work_interactions", 2)
        hospital_params.write_params(constant.TEST_HOSPITAL_FILE)

        file_output = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)

        # get all the people, need to hand case if people having zero connections
        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE,
                               comment = "#", sep = ",", skipinitialspace = True )
        df_indiv = df_indiv.loc[:,[ "ID", "age_group", "occupation_network" ] ]
        df_indiv = pd.merge( df_indiv, ageTypeMap1,
                             left_on = "age_group", right_on = "age_group_1", how = "left" )

        # get all the work connections
        df_int  = pd.read_csv(constant.TEST_INTERACTION_FILE,
                              comment = "#", sep = ",", skipinitialspace = True )
        df_int  = df_int[ df_int["type"] == constant.HOSPITAL_WORK ]
        df_int = pd.merge( df_int, ageTypeMap1,  on = "age_group_1", how = "left" )
        df_int = pd.merge( df_int, ageTypeMap2, on = "age_group_2", how = "left" )

        # get the number of connections for each person
        df_n_int = df_int.groupby( [ "ID_1" ] ).size().reset_index( name = "connections" )
        df_n_int = pd.merge( df_indiv, df_n_int, left_on = "ID", right_on = "ID_1", how = "left" )
        df_n_int.fillna( 0, inplace = True )

        # check that no children or elderly people are in the healthcare worker network.

        n = sum(  ( df_int[ "age_group_1" ]  <= constant.AGE_10_19 )  )
        np.testing.assert_equal( n, 0, "Children (0 - 19) found on healthcare worker network" )

        n = sum(  ( df_int[ "age_group_2" ] <= constant.AGE_10_19 ) )
        np.testing.assert_equal( n, 0, "Children (0 - 19) found on healthcare worker network" )

        n = sum(  ( df_int[ "age_group_1" ] >= constant.AGE_60_69 ) )
        np.testing.assert_equal( n, 0, "Elderly people (60+) found on healthcare worker network" )

        n = sum(  ( df_int[ "age_group_2" ] >= constant.AGE_60_69 ) )
        np.testing.assert_equal( n, 0, "Elderly people (60+) found on healthcare worker network" )


        # check the mean number of networks connections by network
        mean = df_n_int[ df_n_int[ "occupation_network" ] == constant.HOSPITAL_WORK_NETWORK ].loc[:,"connections"].mean()
        np.testing.assert_allclose( mean, 2, rtol = tolerance )

    def test_hcw_network_recurrence( self ):
        """
        Check to see that healthcare workers only meet with the same person
        once per day on the hospital worker network.
        Check to see that when you look over multiple days that
        the mean number of unique contacts is mean_daily/daily_fraction
        """
        tol = 0.02

        end_time = 15
        n_total = 10000
        work_network_rewire = 0.1

        hcw_mean_work_interactions = 2

        hospital_params = ParameterSet(constant.TEST_HOSPITAL_FILE, line_number = 1)
        hospital_params.set_param( "hcw_mean_work_interactions", 2)
        hospital_params.write_params(constant.TEST_HOSPITAL_FILE)

        params = utils.get_params_swig()
        params.set_param( "n_total", n_total )
        params.set_param( "hospital_on", 1 )
        params.set_param( "end_time", end_time )
        params.set_param( "work_network_rewire", work_network_rewire )
        model  = utils.get_model_swig( params )

        # step through time until we need to start to save the interactions each day
        model.one_time_step()
        model.write_interactions_file()
        df_inter = pd.read_csv(constant.TEST_INTERACTION_FILE)
        df_inter[ "time" ] = 0

        # Go over total number of time steps
        for time in range( end_time ):
            model.one_time_step();
            model.write_interactions_file()
            df = pd.read_csv(constant.TEST_INTERACTION_FILE)
            df[ "time" ] = time + 1
            df_inter = pd.concat([df_inter, df])

        # Check if type is hospital work network type
        df_inter = df_inter[ list( df_inter[ "type" ] == constant.HOSPITAL_WORK ) ]

        # check to see there are sufficient daily connections and only one per set of contacts a day
        df_unique_daily = df_inter.groupby( ["time","ID_1","ID_2"]).size().reset_index(name="N");
        min_size = (end_time + 1 ) *  min( 1, hcw_mean_work_interactions )

        np.testing.assert_equal(sum(df_unique_daily["N"] == 1) > min_size, True, "Less contacts than expected on the healthcare worker networks" )
        np.testing.assert_equal(sum(df_unique_daily["N"] != 1), 0, "Repeat connections on same day on the healthcare worker networks" )

        # check the mean unique connections over multiple days is hospital worker connections mean
        df_unique = df_inter.groupby(["occupation_network_1","ID_1","ID_2"]).size().reset_index(name="N_unique")
        df_unique = df_unique.groupby(["occupation_network_1","ID_1"]).size().reset_index(name="N_conn")
        df_unique = df_unique.groupby(["occupation_network_1"]).mean()

        actual   = df_unique.loc[ [constant.HOSPITAL_WORK_NETWORK] ]["N_conn"]
        expected = hcw_mean_work_interactions
        np.testing.assert_allclose(actual,expected,rtol=tol,err_msg="Expected mean unique occupational contacts over multiple days not as expected")