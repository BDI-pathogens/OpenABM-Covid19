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


enum EVENT_TYPES{
	UNINFECTED,
	PRESYMPTOMATIC,
	ASYMPTOMATIC,
	SYMPTOMATIC,
	HOSPITALISED,
	CRITICAL,
	RECOVERED,
	DEATH,
	QUARANTINED,
	QUARANTINE_RELEASE,
	TEST_TAKE,
	TEST_RESULT,
	CASE,
	N_EVENT_TYPES
};

enum TRANSITIONS_TYPES{
	ASYMPTOMATIC_RECOVERED,
	PRESYMPTOMATIC_SYMPTOMATIC,
	SYMPTOMATIC_RECOVERED,
	SYMPTOMATIC_HOSPITALISED,
	HOSPITALISED_CRITICAL,
	HOSPITALISED_RECOVERED,
	CRITICAL_DEATH,
	CRITICAL_RECOVERED,
	SYMPTOMATIC_QUARANTINE,
	TRACED_QUARANTINE,
	TEST_RESULT_QUARANTINE,
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

enum WORK_NETWORKS{
	NETWORK_0_9,
	NETWORK_10_19,
	NETWORK_20_69,
	NETWORK_70_79,
	NETWORK_80,
    //NETWORK_HOSPITAL, //TODO: kelvin change
	N_WORK_NETWORKS
};

enum WORK_NETWORKS_TYPES{
	NETWORK_TYPE_CHILD,
	NETWORK_TYPE_ADULT,
	NETWORK_TYPE_ELDERLY,
	N_WORK_NETWORK_TYPES
};

//TODO: kelvin change
enum WORKER_TYPES {
    DOCTOR,
    NURSE,
    OTHER,
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

enum INTERACTION_TYPE{
	HOUSEHOLD,
	WORK,
	RANDOM,
	N_INTERACTION_TYPES
};

#define UNKNOWN -1
#define NO_EVENT -1
#define NO_EDGE -1
#define NO_TEST -2
#define TEST_ORDERED -1
#define NOT_RECURSIVE 1000
#define MAX_DAILY_INTERACTIONS_KEPT 10
#define MAX_TIME 500
#define MAX_INFECTIOUS_PERIOD 40
#define N_DRAW_LIST 1000
#define INPUT_CHAR_LEN 100

gsl_rng * rng;

#endif /* CONSTANT_H_ */

