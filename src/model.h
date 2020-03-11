/*
 * model.h
 *
 *  Description: Top level model 'object' c
 *  Created on:  5 Mar 2020
 *      Author:  hinchr
 */

#ifndef MODEL_H_
#define MODEL_H_

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/

#include "structure.h"
#include "individual.h"
#include "params.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

struct event_list{
	event *events[MAX_TIME];
	long n_daily[MAX_TIME];
	long n_daily_current[MAX_TIME];
	long n_total;
	long n_current;
	double infectious_curve[MAX_INFECTIOUS_PERIOD];
};

typedef struct{
	parameters *params;
	individual *population;
	int time;

	interaction *interactions;
	long interaction_idx;
	int interaction_day_idx;
	long n_interactions;
	long *possible_interactions;
	long n_possible_interactions;
	long n_total_intereactions;

	event *events;
	event *next_event;

	event_list *presymptomatic;
	event_list *asymptomatic;
	event_list *symptomatic;
	event_list *hospitalised;
	event_list *recovered;
	event_list *death;
	event_list *quarantined;
	event_list *quarantine_release;
	event_list *test_take;
	event_list *test_result;

	int *asymptomatic_time_draws;
	int *symptomatic_time_draws;
	int *hospitalised_time_draws;
	int *recovered_time_draws;
	int *death_time_draws;

	long n_quarantine_days;
} model;

struct event{
	individual *individual;
	event *next;
	event *last;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

model* new_model(parameters *);
void set_up_population( model* );
void set_up_interactions( model* );
void set_up_events( model* );
void set_up_distributions( model* );
void set_up_seed_infection( model* );
void destroy_model( model* );

int one_time_step( model* );
void build_daily_newtork( model* );
void transmit_virus( model* );
void transition_to_symptomatic( model* );
void transition_to_hospitalised( model* );
void transition_to_recovered( model* );
void transition_to_death( model* );
void release_from_quarantine( model* );

event* new_event( model* );
event* add_individual_to_event_list( event_list*, individual*, int, model* );
void set_up_event_list( event_list*, parameters* );
void remove_event_from_event_list( event_list*, event*, model*, int );
void update_event_list_counters( event_list*, model* );

void new_infection( model*, individual*, individual* );

#endif /* MODEL_H_ */
