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
	int age_type;
	int work_network;

	int base_random_interactions;
	int random_interactions;
	int n_interactions[MAX_DAILY_INTERACTIONS_KEPT];
	interaction *interactions[MAX_DAILY_INTERACTIONS_KEPT];

	individual *infector;
	int infector_status;
	int infector_network;

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
	
	trace_token *trace_tokens;
	trace_token *index_trace_token;
	double traced_on_this_trace;

	int app_user;
};

struct interaction{
	int type;
	int traceable;
	individual *individual;
	interaction *next;
};

/************************************************************************/
/******************************  Macros**** *****************************/
/************************************************************************/

#define time_infected( indiv ) ( max( max( indiv->time_event[PRESYMPTOMATIC], indiv->time_event[ASYMPTOMATIC ] ), indiv->time_event[PRESYMPTOMATIC_MILD] ) )
#define is_in_hospital( indiv ) ( ( indiv->status == HOSPITALISED || indiv->status == CRITICAL || indiv->status == HOSPITALISED_RECOVERING ) )

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
void set_hospitalised_recovering( individual*, parameters*, int );
void set_critical( individual*, parameters*, int );
void set_dead( individual*, parameters*, int );
void set_case( individual*, int );
void update_random_interactions( individual*, parameters* );

void destroy_individual( individual* );


#endif /* INDIVIDUAL_H_ */
