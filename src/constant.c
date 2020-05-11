/*
 * constant.c
 *
 *  Created on: 22 Mar 2020
 *      Author: hinchr
 */

#include "constant.h"

const int AGE_OCCUPATION_MAP[N_AGE_GROUPS] = {
	PRIMARY_NETWORK,
	SECONDARY_NETWORK,
	WORKING_NETWORK,
	WORKING_NETWORK,
	WORKING_NETWORK,
	WORKING_NETWORK,
	WORKING_NETWORK,
	RETIRED_NETWORK,
	ELDERLY_NETWORK	
};

const int NETWORK_TYPE_MAP[N_OCCUPATION_NETWORKS] = {
	NETWORK_TYPE_CHILD,
	NETWORK_TYPE_CHILD,
	NETWORK_TYPE_ADULT,
	NETWORK_TYPE_ELDERLY,
	NETWORK_TYPE_ELDERLY
};

const int AGE_TYPE_MAP[N_AGE_GROUPS] = {
	AGE_TYPE_CHILD,
	AGE_TYPE_CHILD,
	AGE_TYPE_ADULT,
	AGE_TYPE_ADULT,
	AGE_TYPE_ADULT,
	AGE_TYPE_ADULT,
	AGE_TYPE_ADULT,
	AGE_TYPE_ELDERLY,
	AGE_TYPE_ELDERLY
};

const char* AGE_TEXT_MAP[N_AGE_GROUPS] = {
	"0-9 years",
	"10-19 years",
	"20-29 years",
	"30-39 years",
	"40-49 years",
	"50-59 years",
	"60-69 years",
	"70-79 years",
	"80+ years"
};

const int EVENT_TYPE_TO_WARD_MAP[N_EVENT_TYPES] = {
	NOT_IN_HOSPITAL,
	NOT_IN_HOSPITAL,
	NOT_IN_HOSPITAL,
	NOT_IN_HOSPITAL,
	NOT_IN_HOSPITAL,
	NOT_IN_HOSPITAL,
	COVID_GENERAL,
	COVID_ICU,
	COVID_GENERAL,
	NOT_IN_HOSPITAL,
	NOT_IN_HOSPITAL,
	NOT_IN_HOSPITAL,
	NOT_IN_HOSPITAL,
	NOT_IN_HOSPITAL,
	NOT_IN_HOSPITAL,
	NOT_IN_HOSPITAL,
	NOT_IN_HOSPITAL,
	NOT_IN_HOSPITAL,
	NOT_IN_HOSPITAL,
	COVID_GENERAL,
	COVID_ICU,
	NOT_IN_HOSPITAL,
	NOT_IN_HOSPITAL,
};

gsl_rng * rng;
