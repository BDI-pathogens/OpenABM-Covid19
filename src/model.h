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
#include "hospital.h"

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
	double **infectious_curve;
};

struct directory{
	long n_idx;
	int *n_jdx;
	long **val;
};

struct model{
	parameters *params;
	individual *population;
	int time;

	interaction_block **interaction_blocks;
	int interaction_day_idx;
	long *possible_interactions;
	long n_possible_interactions;
	long n_total_intereactions;
	long n_occupation_networks;

	network *random_network;
	network *household_network;
	int use_custom_occupation_networks;
	network **occupation_network;
	directory *household_directory;
	network *user_network;
	double mean_interactions;
	double mean_interactions_by_age[ N_AGE_TYPES ];
	int rebuild_networks;

	long manual_trace_interview_quota;
	long manual_trace_notification_quota;

	event_block *event_block;
	event *next_event;
	event_list *event_lists;

	trace_token_block *trace_token_block;
	trace_token *next_trace_token;
	long n_trace_tokens_used;
	long n_trace_tokens;

	int **transition_time_distributions;

	long n_quarantine_days;

	long n_quarantine_app_user;
	long n_quarantine_infected;
	long n_quarantine_recovered;
	long n_quarantine_app_user_infected;
	long n_quarantine_app_user_recovered;
	long n_quarantine_events;
	long n_quarantine_events_app_user;
	long n_quarantine_release_events;
	long n_quarantine_release_events_app_user;

	long n_population_by_age[ N_AGE_GROUPS ];
	long n_vaccinated_fully;
	long n_vaccinated_symptoms;
	long n_vaccinated_fully_by_age[ N_AGE_GROUPS ];
	long n_vaccinated_symptoms_by_age[ N_AGE_GROUPS ];

	hospital *hospitals;
};

struct event_block{
	event *events;
	event_block *next;
};

struct event{
	individual *individual;
	short type;
	short time;
	event *next;
	event *last;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

#define n_current( model, type ) ( model->event_lists[type].n_current )
#define n_total( model, type ) ( model->event_lists[type].n_total )
#define n_total_by_day( model, type, day ) ( model->event_lists[type].n_daily[day] )
#define n_total_age( model, type, age ) ( model->event_lists[type].n_total_by_age[age] )
#define n_daily( model, type, day ) ( model->event_lists[type].n_daily_current[day] )

model* new_model(parameters *);
void set_up_population( model* );
void set_up_healthcare_workers_and_hospitals( model* );
void set_up_interactions( model* );
void set_up_events( model* );
void add_event_block( model* , float );
void set_up_seed_infection( model* );
void set_up_networks( model* );
void set_up_counters( model* );
void reset_counters( model* );
void set_up_occupation_network( model* );
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
void add_interaction_block( model*, long );
void return_interactions( model* );

void add_interactions_from_network( model*, network* );
void build_daily_network( model* );
void build_random_network( model*, network*, long, long* );
void build_random_network_default( model* );
void build_random_network_user( model*, network* );
int add_user_network( model*, int, int, int, int, double, long, long*, long*, char* );
int add_user_network_random( model*, int, int, long, long*, int*, char* );
int delete_network( model*, network*n );
network* get_network_by_id( model*, int );
int get_network_ids( model*, int*, int );
int get_network_id_by_index( model*, int );



#endif /* MODEL_H_ */
