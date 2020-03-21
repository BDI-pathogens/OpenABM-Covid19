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
	long house_no;
	int age_group;
	int work_network;

	int base_random_interactions;
	int random_interactions;
	int n_interactions[MAX_DAILY_INTERACTIONS_KEPT];
	interaction *interactions[MAX_DAILY_INTERACTIONS_KEPT];
	individual *infector;

	int status;
	int is_case;
	double hazard;
	event *current_disease_event;
	event *next_disease_event;
	int *time_event;

	int quarantined;
	event *quarantine_event;
	event *quarantine_release_event;
	int quarantine_test_result;
	
	int app_user;
};

struct interaction{
	int type;
	individual *individual;
	interaction *next;
};

/************************************************************************/
/******************************  Macros**** *****************************/
/************************************************************************/

#define time_infected( indiv ) ( max( indiv->time_event[PRESYMPTOMATIC], indiv->time_event[ASYMPTOMATIC ] ) )

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialize_individual( individual*, parameters*, long );
void initialize_hazard( individual*, parameters* );
void set_age_group( individual*, parameters*, int );
void set_house_no( individual*, long );
void set_quarantine_status( individual*, parameters*, int, int );
void set_recovered( individual*, parameters*, int );
void set_hospitalised( individual*, parameters*, int );
void set_critical( individual*, parameters*, int );
void set_dead( individual*, int );
void set_case( individual*, int );

void destroy_individual( individual* );


#endif /* INDIVIDUAL_H_ */
