"""
Utilities to modify parameters objects associated with testing the COVID19-IBM model

Created: March 2020
Author: p-robot
"""

from parameters import ParameterSet


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
    
    params.set_param("quarantine_household_on_positive", 0)
    params.set_param("quarantine_household_on_symptoms", 0)
    params.set_param("quarantine_household_on_traced", 0)
    params.set_param("quarantine_household_contacts_on_positive", 0)
    params.set_param("quarantined_daily_interactions", 0)
    
    params.set_param("app_users_fraction", 0.0)
    
    # Set interventions associated with a time to be beyond the end of the simulation
    params.set_param("app_turn_on_time", end_time)
    params.set_param("social_distancing_time_on", end_time + 1)
    params.set_param("social_distancing_time_off", end_time + 2)
    params.set_param("testing_symptoms_time_on", end_time + 1)
    params.set_param("testing_symptoms_time_off", end_time + 2)
    
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
    params.set_param("quarantine_household_on_traced", 0)
    params.set_param("quarantine_household_contacts_on_positive", 0)
    params.set_param("self_quarantine_fraction", 0.0)
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
