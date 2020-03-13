/*
 * individual.h
 *
 *  Created on: 5 Mar 2020
 *      Author: hinchr
 */

#ifndef INDIVIDUAL_H_
#define INDIVIDUAL_H_

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/

#include <gsl/gsl_rng.h>
#include "structure.h"
#include "params.h"
#include "constant.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct individual individual;

struct individual{
	long idx;
	int status;

	int age_group;
	int work_network;

	int quarantined;
	int base_random_interactions;
	int random_interactions;
	double hazard;
	int n_interactions[MAX_DAILY_INTERACTIONS_KEPT];
	interaction *interactions[MAX_DAILY_INTERACTIONS_KEPT];
	individual *infector;

	int time_infected;
	int time_symptomatic;
	int time_asymptomatic;
	int time_hospitalised;
	int time_death;
	int time_recovered;
	int time_quarantined;

	event *current_disease_event;
	int next_disease_type;

	event *quarantine_event;
	int quarantine_test_result;
	
	int app_user;
};

struct interaction{
	individual *individual;
	interaction *next;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialize_individual( individual*, parameters*, long );
void set_age_group( individual*, parameters*, int );
void set_quarantine_status( individual*, parameters*, int, int );
void set_recovered( individual*, parameters*, int );
void set_hospitalised( individual*, parameters*, int );
void set_dead( individual*, int );

void destroy_individual( individual* );


#endif /* INDIVIDUAL_H_ */
