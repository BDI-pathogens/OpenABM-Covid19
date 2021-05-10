/*
 * constant.h
 *
 *  Created on: 5 Mar 2020
 *      Author: hinchr
 */

#ifndef CONSTANT_H_
#define CONSTANT_H_

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

#define FALSE 0
#define TRUE 1
#define ERROR -1

enum EVENT_TYPES{
	SUSCEPTIBLE,
	PRESYMPTOMATIC,
	PRESYMPTOMATIC_MILD,
	ASYMPTOMATIC,
	SYMPTOMATIC,
	SYMPTOMATIC_MILD,
	HOSPITALISED,
	CRITICAL,
	HOSPITALISED_RECOVERING,
	RECOVERED,
	DEATH,
	QUARANTINED,
	QUARANTINE_RELEASE,
	TEST_TAKE,
	TEST_RESULT,
	CASE,
	TRACE_TOKEN_RELEASE,
	NOT_IN_HOSPITAL,
	WAITING,
	GENERAL,
	ICU,
	MORTUARY,
	DISCHARGED,
	MANUAL_CONTACT_TRACING,
	TRANSITION_TO_HOSPITAL,
	TRANSITION_TO_CRITICAL,
	VACCINE_PROTECT,
	VACCINE_WANE,
	N_EVENT_TYPES
};
#define N_NEWLY_INFECTED_STATES 3

enum TRANSITIONS_TYPES{
	ASYMPTOMATIC_RECOVERED,
	PRESYMPTOMATIC_SYMPTOMATIC,
	PRESYMPTOMATIC_MILD_SYMPTOMATIC_MILD,
	SYMPTOMATIC_RECOVERED,
	SYMPTOMATIC_HOSPITALISED,
	SYMPTOMATIC_MILD_RECOVERED,
	HOSPITALISED_CRITICAL,
	HOSPITALISED_RECOVERED,
	CRITICAL_DEATH,
	CRITICAL_HOSPITALISED_RECOVERING,
	HOSPITALISED_RECOVERING_RECOVERED,
	SYMPTOMATIC_QUARANTINE,
	TRACED_QUARANTINE_SYMPTOMS,
	TRACED_QUARANTINE_POSITIVE,
	TEST_RESULT_QUARANTINE,
	RECOVERED_SUSCEPTIBLE,
	VACCINE_PROTECTED_SUSCEPTIBLE,
	VACCINE_PROTECTED_SYMPTOMS_VACCINE_WANED,
	N_TRANSITION_TYPES
};

enum AGE_GROUPS{
	AGE_0_9,
	AGE_10_19,
	AGE_20_29,
	AGE_30_39,
	AGE_40_49,
	AGE_50_59,
	AGE_60_69,
	AGE_70_79,
	AGE_80,
	N_AGE_GROUPS
};

enum AGE_TYPES{
	AGE_TYPE_CHILD,
	AGE_TYPE_ADULT,
	AGE_TYPE_ELDERLY,
	N_AGE_TYPES
};

enum OCCUPATION_NETWORKS{
	PRIMARY_NETWORK,
	SECONDARY_NETWORK,
	WORKING_NETWORK,
	RETIRED_NETWORK,
	ELDERLY_NETWORK,
	N_DEFAULT_OCCUPATION_NETWORKS
};

enum OCCUPATION_NETWORKS_TYPES{
	NETWORK_TYPE_CHILD,
	NETWORK_TYPE_ADULT,
	NETWORK_TYPE_ELDERLY,
	N_OCCUPATION_NETWORK_TYPES
};

enum WORKER_TYPES {
    DOCTOR,
    NURSE,
    N_WORKER_TYPES
};

enum DEFAULT_NETWORKS{
	HOUSEHOLD_NETWORK,
	OCCUPATION_PRIMARY_NETWORK,
	OCCUPATION_SECONDARY_NETWORK,
	OCCUPATION_WORKING_NETWORK,
	OCCUPATION_RETIRED_NETWORK,
	OCCUPATION_ELDERLY_NETWORK,
	RANDOM_NETWORK,
	N_DEFAULT_NETWORKS
};

enum NETWORK_CONSTRUCTIONS{
	NETWORK_CONSTRUCTION_BESPOKE,
	NETWORK_CONSTRUCTION_HOUSEHOLD,
	NETWORK_CONSTRUCTION_WATTS_STROGATZ,
	NETWORK_CONSTRUCTION_RANDOM_DEFAULT,
	NETWORK_CONSTRUCTION_RANDOM,
	N_NETWORK_CONSTRUCTIONS
};

extern const int AGE_OCCUPATION_MAP[N_AGE_GROUPS];
extern const int NETWORK_TYPE_MAP[N_DEFAULT_OCCUPATION_NETWORKS];
extern const int OCCUPATION_DEFAULT_MAP[N_DEFAULT_OCCUPATION_NETWORKS];
extern const int AGE_TYPE_MAP[N_AGE_GROUPS];
extern const char* AGE_TEXT_MAP[N_AGE_GROUPS];
extern const int EVENT_TYPE_TO_WARD_MAP[N_EVENT_TYPES];
extern const char* DEFAULT_NETWORKS_NAMES[N_DEFAULT_NETWORKS];
extern const int NEWLY_INFECTED_STATES[N_NEWLY_INFECTED_STATES];

enum HOUSEHOLD_SIZE{
	HH_1,
	HH_2,
	HH_3,
	HH_4,
	HH_5,
	HH_6,
	N_HOUSEHOLD_MAX
};

enum INTERACTION_TYPE{
	HOUSEHOLD,
	OCCUPATION,
	RANDOM,
	HOSPITAL_WORK,
	HOSPITAL_DOCTOR_PATIENT_GENERAL,
	HOSPITAL_NURSE_PATIENT_GENERAL,
	HOSPITAL_DOCTOR_PATIENT_ICU,
	HOSPITAL_NURSE_PATIENT_ICU,
	N_INTERACTION_TYPES
};

enum DISTRIBUTIONS{
	FIXED,
	NEGATIVE_BINOMIAL
};

enum HOSPITAL_WARD_TYPES{
    COVID_GENERAL,
    COVID_ICU,
    N_HOSPITAL_WARD_TYPES
};

enum INDEX_STATUS{
	SYMPTOMS_ONLY,
	POSITIVE_TEST
};

enum QUARANTINE_REASONS{
    QR_SELF_POSITIVE,
	QR_SELF_SYMPTOMS,
	QR_HOUSEHOLD_POSITIVE,
	QR_HOUSEHOLD_SYMPTOMS,
	QR_TRACE_POSITIVE,
	QR_TRACE_SYMPTOMS,
	N_QUARANTINE_REASONS
};

enum TRACE_TYPE{
	NO_TRACE,
	DIGITAL_TRACE,
	MANUAL_TRACE
};

enum VACCINE_STATUS{
	NO_VACCINE,
	VACCINE_NO_PROTECTION,
	VACCINE_PROTECTED_FULLY,
	VACCINE_PROTECTED_SYMPTOMS,
//	VACCINE_BOOSTED_NO_PROTECTION,
//	VACCINE_BOOSTED_PROTECTED_FULLY,
//	VACCINE_BOOSTED_PROTECTED_SYMPTOMS,
	VACCINE_WANED
};

enum VACCINE_TYPES{
	VACCINE_TYPE_FULL,
	VACCINE_TYPE_SYMPTOMS
};

#define UNKNOWN -1
#define NO_PRIORITY_TEST -1
#define NO_EVENT -1
#define NO_EDGE -1
#define NO_TEST -2
#define TEST_ORDERED -1
#define TEST_ORDERED_PRIORITY -3
#define NOT_RECURSIVE 1000
#define MAX_DAILY_INTERACTIONS_KEPT 10
#define MAX_TIME 500
#define MAX_INFECTIOUS_PERIOD 200
#define N_DRAW_LIST 1000
#define INPUT_CHAR_LEN 300

#define NO_WARD -1
#define NOT_HEALTHCARE_WORKER -1
#define NO_HOSPITAL -1
#define HOSPITAL_WORK_NETWORK -1
#define N_HOSPITAL_INTERACTION_TYPES 5

#define MAX_N_STRAINS 20000
#define ANTIGEN_PHEN_DIM 2

extern gsl_rng * rng;

#endif /* CONSTANT_H_ */
