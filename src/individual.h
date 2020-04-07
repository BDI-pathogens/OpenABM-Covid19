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

    int app_user;

    int ward_idx;
    int ward_type;
    int hospital_idx;

    int hospital_state;
	event *current_hospital_event;
	event *next_hospital_event;

    int worker_type;
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

#define time_infected( indiv ) ( max( indiv->time_event[PRESYMPTOMATIC], indiv->time_event[ASYMPTOMATIC ] ) )
#define is_in_hospital( indiv ) ( ( indiv->status == HOSPITALISED || indiv->status == CRITICAL ) )

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
void set_dead( individual*, parameters*, int );
void set_case( individual*, int );
void set_waiting( individual*, parameters*, int );
void set_general_admission( individual*, parameters*, int );
void set_icu_admission( individual*, parameters*, int );
void set_mortuary_admission( individual*, parameters*, int );
void set_discharged( individual*, parameters*, int );
void update_random_interactions( individual*, parameters* );

void destroy_individual( individual* );


#endif /* INDIVIDUAL_H_ */
