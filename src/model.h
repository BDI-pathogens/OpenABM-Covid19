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

#include "individual.h"
#include "params.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct event event;
typedef struct event_list event_list;

struct event_list{
	event *events[MAX_TIME];
	long n_daily[MAX_TIME];
	long n_total;
	long n_current;
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

	event *events;
	long event_idx;

	event_list infected;
	double infectious_curve[MAX_INFECTIOUS_PERIOD];

	event_list symptomatic;
	double symptomatic_curve[MAX_INFECTIOUS_PERIOD];

} model;

struct event{
	individual *individual;
	event *next;
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

event* new_event( model* );
void new_infection( model*, individual* );

#endif /* MODEL_H_ */
