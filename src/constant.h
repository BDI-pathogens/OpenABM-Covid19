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

//TODO: add hospitalised type event
enum EVENT_TYPES{
	UNINFECTED,
	PRESYMPTOMATIC,
	PRESYMPTOMATIC_MILD,
	ASYMPTOMATIC,
	SYMPTOMATIC,
	SYMPTOMATIC_MILD,
    HOSPITALISED, //severe TODO: change this to hospitalisation required!! might be added to waiting list before hand
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
	NOT_IN_HOSPITAL, //TOM: Events for hospital states.
	WAITING,
	GENERAL,
	ICU,
    MORTUARY,
    DISCHARGED,
	N_EVENT_TYPES
};
//for transition time curves
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
	TRACED_QUARANTINE,
	TEST_RESULT_QUARANTINE,
	HOSPITAL_TRANSITION,    //TOM: Event transitions for all hospital states.
	N_TRANSITION_TYPES      // Added to params: mean_time_hospital transition (1), sd_time_hospital_transition (0).
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

enum WORK_NETWORKS{
    NETWORK_0_9, // healthcare workers children still in school (intervention where only key workers can have kids at school)
    NETWORK_10_19,
    NETWORK_20_69, // pick certain number from this network to be healthcare worker
	NETWORK_70_79,
	NETWORK_80,
	N_WORK_NETWORKS
};

enum WORK_NETWORKS_TYPES{
	NETWORK_TYPE_CHILD,
	NETWORK_TYPE_ADULT,
	NETWORK_TYPE_ELDERLY,
	N_WORK_NETWORK_TYPES
};

//TODO: change to HOSPITAL_WORKER_TYPES
enum WORKER_TYPES {
    DOCTOR,
    NURSE,
    N_WORKER_TYPES
};

extern const int AGE_WORK_MAP[N_AGE_GROUPS];
extern const int NETWORK_TYPE_MAP[N_WORK_NETWORKS];
extern const int AGE_TYPE_MAP[N_AGE_GROUPS];
extern const char* AGE_TEXT_MAP[N_AGE_GROUPS];



enum HOUSEHOLD_SIZE{
	HH_1,
	HH_2,
	HH_3,
	HH_4,
	HH_5,
	HH_6,
	N_HOUSEHOLD_MAX
};

//DONE: Add hospital interaction type for workers and patients.
//TODO: ALTER INFECTIVITY OF HOSPITAL INTERACTION TYPES.
enum INTERACTION_TYPE{
	HOUSEHOLD,
	WORK,
	RANDOM,
    HOSPITAL_WORK, //Interactions between healthcare workers
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

#define UNKNOWN -1
#define NO_EVENT -1
#define NO_EDGE -1
#define NO_TEST -2
#define TEST_ORDERED -1
#define NOT_RECURSIVE 1000
#define MAX_DAILY_INTERACTIONS_KEPT 10
#define MAX_TIME 500
#define MAX_INFECTIOUS_PERIOD 200
#define N_DRAW_LIST 1000
#define INPUT_CHAR_LEN 300

#define NO_WARD -1
#define NOT_HEALTHCARE_WORKER -1
#define NO_HOSPITAL -1

extern gsl_rng * rng;

#endif /* CONSTANT_H_ */

