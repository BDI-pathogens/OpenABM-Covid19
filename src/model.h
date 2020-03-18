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
#include "network.h"
#include "params.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

struct event_list{
	int type;
	event **events;
	long *n_daily;
	long **n_daily_by_age;
	long *n_daily_current;
	long n_total;
	long *n_total_by_age;
	long n_current;
	double *infectious_curve;
};

struct model{
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

	network *random_network;
	network *household_network;
	network **work_network;

	event *events;
	event *next_event;
	event_list *event_lists;

	int **transition_time_distributions;

	long n_quarantine_days;
};

struct event{
	individual *individual;
	int type;
	int time;
	event *next;
	event *last;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

#define n_current( model, type ) ( model->event_lists[type].n_current )
#define n_total( model, type ) ( model->event_lists[type].n_total )
#define n_total_age( model, type, age ) ( model->event_lists[type].n_total_by_age[age] )
#define n_daily( model, type, day ) ( model->event_lists[type].n_daily_current[day] )

model* new_model(parameters *);
void set_up_population( model* );
void set_up_interactions( model* );
void set_up_events( model* );
void set_up_distributions( model* );
void set_up_seed_infection( model* );
void set_up_networks( model* );
void set_up_work_network( model*, int );
void set_up_app_users( model* );
void set_up_individual_hazard( model* );
void destroy_model( model* );

int one_time_step( model* );
void flu_infections( model* );

event* new_event( model* );
event* add_individual_to_event_list( model*, int, individual*, int );
void set_up_event_list( model*, parameters*, int );
void destroy_event_list( model*, int );
void remove_event_from_event_list( model*, event* );
void update_event_list_counters(  model*, int );
void transition_events( model*, int, void( model*, individual* ), int );

void add_interactions_from_network( model*, network*, int, int, double );
void build_daily_newtork( model* );
void build_random_network( model * );
void build_household_network( model * );

#endif /* MODEL_H_ */
