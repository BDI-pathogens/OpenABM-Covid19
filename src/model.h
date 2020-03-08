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
	parameters params;
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
	long event_idx;

	event_list infected;
	event_list asymptomatic;
	event_list symptomatic;
	event_list hospitalised;
	event_list recovered;
	event_list death;

	int asymptomatic_time_draws[N_DRAW_LIST];
	int symptomatic_time_draws[N_DRAW_LIST];
	int hospitalised_time_draws[N_DRAW_LIST];
	int recovered_time_draws[N_DRAW_LIST];
	int death_time_draws[N_DRAW_LIST];


} model;

struct event{
	individual *individual;
	event *next;
	event *last;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

model* new_model();
void set_up_population( model* );
void set_up_interactions( model* );
void set_up_events( model* );
void set_up_distributions( model* );
void set_up_seed_infection( model* );
void destroy_model( model* );

int one_time_step( model* );
void build_daily_newtork( model* );
void transmit_virus( model* );
void transition_infected( model* );
void transition_symptomatic( model* );
void transition_hospitalised( model* );

event* new_event( model* );
event* add_individual_to_event_list( event_list*, individual*, int, model* );
void remove_event_from_event_list( event_list*, event*, int );
void update_event_list_counters( event_list*, model* );

void new_infection( model*, individual* );

#endif /* MODEL_H_ */
