"""
Utilities to modify parameters objects associated with testing the COVID19-IBM model

Created: March 2020
Author: p-robot
"""
import sys

sys.path.append("src/COVID19")

from parameters import ParameterSet
from COVID19.model import Model, Parameters, ModelParameterException
from . import constant

def turn_off_interventions(params, end_time):
    """
    Function to turn off all interventions and return the same object
    """

    params.set_param("test_on_traced", 0)
    params.set_param("test_on_symptoms", 0)
    params.set_param("quarantine_on_traced", 0)
    params.set_param("traceable_interaction_fraction", 0.0)
    params.set_param("tracing_network_depth", 0)
    params.set_param("allow_clinical_diagnosis", 0)
    params.set_param("self_quarantine_fraction", 0.0)
    params.set_param("manual_trace_on", 0)

    params.set_param("quarantine_household_on_positive", 0)
    params.set_param("quarantine_household_on_symptoms", 0)
    params.set_param("quarantine_household_on_traced_positive", 0)
    params.set_param("quarantine_household_on_traced_symptoms", 0)
    params.set_param("quarantine_household_contacts_on_positive", 0)
    params.set_param("quarantined_daily_interactions", 0)

    params.set_param("app_users_fraction_0_9", 0.0)
    params.set_param("app_users_fraction_10_19", 0.0)
    params.set_param("app_users_fraction_20_29", 0.0)
    params.set_param("app_users_fraction_30_39", 0.0)
    params.set_param("app_users_fraction_40_49", 0.0)
    params.set_param("app_users_fraction_50_59", 0.0)
    params.set_param("app_users_fraction_60_69", 0.0)
    params.set_param("app_users_fraction_70_79", 0.0)
    params.set_param("app_users_fraction_80", 0.0)

    # Set interventions associated with a time to be beyond the end of the simulation
    params.set_param("app_turn_on_time", end_time)
    params.set_param("lockdown_time_on", end_time + 1)
    params.set_param("lockdown_time_off", end_time + 2)
    params.set_param("testing_symptoms_time_on", end_time + 1)
    params.set_param("testing_symptoms_time_off", end_time + 2)
    params.set_param("manual_trace_time_on", end_time)

    return(params)


def turn_off_testing(params):
    """
    Intervention to turn off all testing parameters (so there should be no testing)
    """
    params.set_param("test_on_traced", 0)
    params.set_param("test_on_symptoms", 0)
    return(params)


def turn_off_quarantine(params):
    """
    Turn off all quarantine flags (so there should be nobody quarantined)
    """
    params.set_param("quarantine_on_traced", 0)
    params.set_param("quarantine_household_on_positive", 0)
    params.set_param("quarantine_household_on_symptoms", 0)
    params.set_param("quarantine_household_on_traced_positive", 0)
    params.set_param("quarantine_household_on_traced_symptoms", 0)
    params.set_param("quarantine_household_contacts_on_positive", 0)
    params.set_param("quarantine_household_contacts_on_symptoms", 0)
    params.set_param("self_quarantine_fraction", 0.0)
    return(params)


def set_fatality_fraction_all(params, fraction):
    """
    Set critical fraction to `fraction` for all ages
    """
    param_names = ["fatality_fraction_0_9", "fatality_fraction_10_19", "fatality_fraction_20_29",
        "fatality_fraction_30_39", "fatality_fraction_40_49", "fatality_fraction_50_59",
        "fatality_fraction_60_69", "fatality_fraction_70_79", "fatality_fraction_80"]

    for p in param_names:
        params.set_param(p, fraction)

    return(params)


def set_fraction_asymptomatic_all(params, fraction):
    """
    Set fraction asymptomatic to `fraction` for all ages
    """
    param_names = ["fraction_asymptomatic_0_9", "fraction_asymptomatic_10_19", "fraction_asymptomatic_20_29",
        "fraction_asymptomatic_30_39", "fraction_asymptomatic_40_49", "fraction_asymptomatic_50_59",
        "fraction_asymptomatic_60_69", "fraction_asymptomatic_70_79", "fraction_asymptomatic_80"]

    for p in param_names:
        params.set_param(p, fraction)

    return(params)

def set_location_death_icu_all(params, fraction):
    """
    Set location_death_icu to the same for all ages
    """
    param_names = ["location_death_icu_0_9", "location_death_icu_10_19", "location_death_icu_20_29",
        "location_death_icu_30_39", "location_death_icu_40_49", "location_death_icu_50_59",
        "location_death_icu_60_69", "location_death_icu_70_79", "location_death_icu_80"]

    for p in param_names:
        params.set_param(p, fraction)

    return(params)

def set_critical_fraction_all(params, fraction):
    """
    Set critical fraction to `fraction` for all ages
    """
    param_names = ["critical_fraction_0_9", "critical_fraction_10_19", "critical_fraction_20_29",
        "critical_fraction_30_39", "critical_fraction_40_49", "critical_fraction_50_59",
        "critical_fraction_60_69", "critical_fraction_70_79", "critical_fraction_80"]

    for p in param_names:
        params.set_param(p, fraction)

    return(params)

def set_app_users_fraction_all(params, fraction):
    """
    Set app_users_fraction for all ages
    """
    param_names = [ "app_users_fraction_0_9", "app_users_fraction_10_19",  "app_users_fraction_20_29",
        "app_users_fraction_30_39",  "app_users_fraction_40_49", "app_users_fraction_50_59",
        "app_users_fraction_60_69",  "app_users_fraction_70_79", "app_users_fraction_80" ]

    for p in param_names:
        params.set_param(p, fraction)

    return(params)

def set_priority_test_contacts_all(params, n):
    """
    Set priority_test_contacts for all ages
    """
    param_names = [ "priority_test_contacts_0_9", "priority_test_contacts_10_19",  "priority_test_contacts_20_29",
        "priority_test_contacts_30_39",  "priority_test_contacts_40_49", "priority_test_contacts_50_59",
        "priority_test_contacts_60_69",  "priority_test_contacts_70_79", "priority_test_contacts_80" ]

    for p in param_names:
        params.set_param(p, n)

    return(params)

def set_hospitalisation_fraction_all(params, fraction):
    """
    Set hospitalised fraction to `fraction` for all ages
    """
    param_names = ["hospitalised_fraction_0_9", "hospitalised_fraction_10_19",
        "hospitalised_fraction_20_29", "hospitalised_fraction_30_39",
        "hospitalised_fraction_40_49", "hospitalised_fraction_50_59",
        "hospitalised_fraction_60_69", "hospitalised_fraction_70_79",
        "hospitalised_fraction_80"]

    for p in param_names:
        params.set_param(p, fraction)

    return(params)

def set_relative_susceptibility_equal(params):
    """
    Set all people equally susceptible
    """
    param_names = [
        "relative_susceptibility_0_9",    "relative_susceptibility_10_19",
        "relative_susceptibility_20_29",  "relative_susceptibility_30_39",
        "relative_susceptibility_40_49",  "relative_susceptibility_50_59",
        "relative_susceptibility_60_69",  "relative_susceptibility_70_79",
        "relative_susceptibility_80",
    ]

    for p in param_names:
        params.set_param(p, 1.0)

    return(params)

def set_work_connections_all(params,connections):
    """
    Set the same number of work connections for everyone
    """
    params.set_param( "mean_work_interactions_child", connections )
    params.set_param( "mean_work_interactions_adult", connections )
    params.set_param( "mean_work_interactions_elderly", connections )

    return(params)

def set_random_connections_all(params,connections,sd):
    """
    Set the same number of random connections for everyone
    """
    params.set_param( "mean_random_interactions_child", connections )
    params.set_param( "mean_random_interactions_adult", connections )
    params.set_param( "mean_random_interactions_elderly", connections )
    params.set_param( "sd_random_interactions_child", sd )
    params.set_param( "sd_random_interactions_adult", sd )
    params.set_param( "sd_random_interactions_elderly", sd )

    # zero sd is achieved by having the fixed distribution
    if sd == 0 :
        params.set_param( "random_interaction_distribution", 0 );

    return(params)

def set_homogeneous_random_network_only(params,connections,end_time):
    """
    Set a simple model with a homogeneous population and only
    disease transmission on the random network where all people
    have equal numbers of interactions
    """

    params.set_param( "end_time", end_time)
    params.set_param( "relative_transmission_household", 0.0 )
    params.set_param( "relative_transmission_occupation", 0.0 )
    params.set_param( "relative_transmission_random", 1.0 )
    params.set_param( "mean_time_to_symptoms", end_time + 10 )
    params.set_param( "sd_time_to_symptoms", 2.0 )
    params.set_param( "mean_asymptomatic_to_recovery", end_time + 10 )
    params.set_param( "sd_asymptomatic_to_recovery", 2.0 )

    params = turn_off_interventions(params, end_time)
    params = set_fraction_asymptomatic_all( params, 0.0 )
    params = set_relative_susceptibility_equal(params)
    params = set_random_connections_all(params,connections,0)
    params = set_work_connections_all(params,0)

    return(params)

def get_params_custom():
    """
    Gets the Custom test parameters object
    """
    return Parameters(
        constant.TEST_DATA_FILE,
        1,
        constant.DATA_DIR_TEST,
        constant.TEST_HOUSEHOLD_TEMPLATE,
        constant.TEST_HOSPITAL_FILE,
        1,
        True,
        True
    )

def get_params_swig():
    """
    Gets the Swig parameters object
    """
    return Parameters(
        constant.TEST_DATA_TEMPLATE,
        1,
        constant.DATA_DIR_TEST,
        constant.TEST_HOUSEHOLD_TEMPLATE,
        constant.TEST_HOSPITAL_FILE,
        1,
        True,
        True
    )


def get_model_swig( params ):
    """
    Gets the Swig model object
    """
    return Model( params )

""
