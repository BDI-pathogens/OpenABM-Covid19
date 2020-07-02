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
	int occupation_network;

	int base_random_interactions;
	int random_interactions;
	int n_interactions[MAX_DAILY_INTERACTIONS_KEPT];
	interaction *interactions[MAX_DAILY_INTERACTIONS_KEPT];

	int status;
	double hazard;
	event *current_disease_event;
	event *next_disease_event;
	infection_event *infection_events;

	int quarantined;
	event *quarantine_event;
	event *quarantine_release_event;
	int quarantine_test_result;
	
	trace_token *trace_tokens;
	trace_token *index_trace_token;
	event *index_token_release_event;
	double traced_on_this_trace;

	int app_user;

	int ward_idx;
	int ward_type;

	int hospital_idx;

	int hospital_state;
	int disease_progression_predicted[N_HOSPITAL_WARD_TYPES];
	event *current_hospital_event;
	event *next_hospital_event;

	int worker_type;

};

struct interaction{
	int type;
	int traceable;
	int manual_traceable;
	individual *individual;
	interaction *next;
};

struct interaction_block{
	interaction *interactions;
	interaction_block *next;
};

struct infection_event{
	int *times;
	individual *infector;
	int infector_status;
	int infector_hospital_state;
	int infector_network;
	int time_infected_infector;
	infection_event *next;
	int is_case;
};

/************************************************************************/
/******************************  Macros**** *****************************/
/************************************************************************/

#define time_symptomatic( indiv ) ( max( indiv->infection_events->times[SYMPTOMATIC], indiv->infection_events->times[SYMPTOMATIC_MILD] ) )
#define time_infected( indiv ) ( max( max( indiv->infection_events->times[PRESYMPTOMATIC], indiv->infection_events->times[ASYMPTOMATIC ] ), indiv->infection_events->times[PRESYMPTOMATIC_MILD] ) )
#define time_infected_infection_event( infection_event ) ( max( max( infection_event->times[PRESYMPTOMATIC], infection_event->times[ASYMPTOMATIC ] ), infection_event->times[PRESYMPTOMATIC_MILD] ) )

#define is_in_hospital( indiv ) ( ( indiv->status == HOSPITALISED || indiv->status == CRITICAL || indiv->status == HOSPITALISED_RECOVERING ) )

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialize_individual( individual*, parameters*, long );
void initialize_hazard( individual*, parameters* );
void set_age_group( individual*, parameters*, int );
void set_house_no( individual*, long );
void set_quarantine_status( individual*, parameters*, int, int, model* );
void set_recovered( individual*, parameters*, int , model *);
void set_hospitalised( individual*, parameters*, int );
void set_hospitalised_recovering( individual*, parameters*, int );
void set_critical( individual*, parameters*, int );
void set_dead( individual*, parameters*, int );
void set_case( individual*, int );
void set_waiting( individual*, parameters*, int );
void set_general_admission( individual*, parameters*, int );
void set_icu_admission( individual*, parameters*, int );
void set_mortuary_admission( individual*, parameters*, int );
void set_discharged( individual*, parameters*, int );
void update_random_interactions( individual*, parameters* );
int count_infection_events( individual * );
void destroy_individual( individual* );
void print_individual( model *, long );

#endif /* INDIVIDUAL_H_ */
