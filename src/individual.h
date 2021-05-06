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
	short age_group;
	short age_type;
	short occupation_network;

	short base_random_interactions;
	short random_interactions;
	short *n_interactions;
	interaction **interactions;

	short status;
	float hazard;
	float infectiousness_multiplier;
	event *current_disease_event;
	event *next_disease_event;
	infection_event *infection_events;

	short quarantined;
	event *quarantine_event;
	event *quarantine_release_event;
	short quarantine_test_result;
	
	trace_token *trace_tokens;
	trace_token *index_trace_token;
	event *index_token_release_event;
	float traced_on_this_trace;

	short app_user;

	short ward_idx;
	short ward_type;

	short hospital_idx;

	short hospital_state;
	short disease_progression_predicted[N_HOSPITAL_WARD_TYPES];
	event *current_hospital_event;
	event *next_hospital_event;

	short worker_type;

	short vaccine_status;
	short vaccine_status_next;
	event *vaccine_wane_event;
};

struct interaction{
	short type;
	short network_id;
	short traceable;
	short manual_traceable;
	individual *individual;
	interaction *next;
};

struct interaction_block{
	interaction *interactions;
	long n_interactions;
	long idx;
	interaction_block *next;
};

struct infection_event{
	short *times;
	individual *infector;
	short infector_status;
	short infector_hospital_state;
	short infector_network;
	short time_infected_infector;
	infection_event *next;
	short is_case;
	short network_id;
	float strain_multiplier;
};

/************************************************************************/
/******************************  Macros**** *****************************/
/************************************************************************/

#define time_symptomatic( indiv ) ( max( indiv->infection_events->times[SYMPTOMATIC], indiv->infection_events->times[SYMPTOMATIC_MILD] ) )
#define time_infected( indiv ) ( max( max( indiv->infection_events->times[PRESYMPTOMATIC], indiv->infection_events->times[ASYMPTOMATIC ] ), indiv->infection_events->times[PRESYMPTOMATIC_MILD] ) )
#define time_infected_infection_event( infection_event ) ( max( max( infection_event->times[PRESYMPTOMATIC], infection_event->times[ASYMPTOMATIC ] ), infection_event->times[PRESYMPTOMATIC_MILD] ) )

#define is_in_hospital( indiv ) ( ( indiv->status == HOSPITALISED || indiv->status == CRITICAL || indiv->status == HOSPITALISED_RECOVERING ) )
#define not_in_hospital( indiv ) ( (indiv->hospital_state == NOT_IN_HOSPITAL) || (indiv->hospital_state == DISCHARGED) )

#define vaccine_protected( indiv ) ( (indiv->vaccine_status == VACCINE_PROTECTED_FULLY) || (indiv->vaccine_status == VACCINE_PROTECTED_SYMPTOMS ) )

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void initialize_individual( individual*, parameters*, long );
void initialize_hazard( individual*, parameters* );
void set_age_group( individual*, parameters*, int );
void set_house_no( individual*, long );
void set_quarantine_status( individual*, parameters*, int, int, model* );
void set_recovered( individual*, parameters*, int , model *);
void set_susceptible( individual*, parameters*, int );
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
void set_vaccine_status( individual*, short, short );
void transition_vaccine_status( individual* );
void update_random_interactions( individual*, parameters* );
int count_infection_events( individual * );
void destroy_individual( individual* );
void print_individual( model *, long );

#endif /* INDIVIDUAL_H_ */
