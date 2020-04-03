/*
 * interventions.h
 *
 *  Created on: 18 Mar 2020
 *      Author: hinchr
 */

#ifndef INTERVENTIONS_H_
#define INTERVENTIONS_H_

#include "structure.h"
#include "individual.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/
struct trace_token{
	individual *individual;
	trace_token *next_index;
	trace_token *next;
	trace_token *last;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void set_up_transition_times_intervention( model* );
void set_up_app_users( model*, double );
void update_intervention_policy( model*, int );

void set_up_trace_tokens( model* );
trace_token* new_trace_token( model* );
trace_token* index_trace_token( model*, individual* );

void intervention_quarantine_until( model*, individual*, int, int, trace_token* );
void intervention_quarantine_release( model*, individual* );
void intervention_quarantine_household( model*, individual*, int, int, trace_token* );
void intervention_test_take( model*, individual* );
void intervention_test_result( model*, individual* );
void intervention_notify_contacts( model*, individual*, int, trace_token* );

void intervention_on_symptoms( model*, individual* );
void intervention_on_hospitalised( model*, individual* );
void intervention_on_critical( model*, individual* );
void intervention_on_positive_result( model*, individual* );
void intervention_on_traced( model*, individual*, int, int, trace_token* );

#endif /* INTERVENTIONS_H_ */
