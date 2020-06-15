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
	trace_token *last_index;
	trace_token *next;
	trace_token *last;
	int contact_time;
	int index_status;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void set_up_transition_times_intervention( model* );
void set_up_app_users( model* );
void set_up_risk_scores( model* );
void destroy_risk_scores( model* );
void update_intervention_policy( model*, int );

void set_up_trace_tokens( model* );
trace_token* new_trace_token( model*, individual*, int );
trace_token* index_trace_token( model*, individual* );
void remove_traced_on_this_trace( model*, individual* );
void remove_one_trace_token_from_index( model*, trace_token* );
void remove_traces_on_individual( model*, individual* );
void intervention_trace_token_release( model*, individual* );

int intervention_quarantine_until( model*, individual*, int, int, trace_token*, int, double );
void intervention_quarantine_release( model*, individual* );
void intervention_quarantine_household( model*, individual*, int, int, trace_token*, int );
void intervention_test_take( model*, individual* );
void intervention_test_result( model*, individual* );
void intervention_notify_contacts( model*, individual*, int, trace_token*, int );
void intervention_index_case_symptoms_to_positive( model*, trace_token* );

void intervention_on_symptoms( model*, individual* );
void intervention_on_hospitalised( model*, individual* );
void intervention_on_critical( model*, individual* );
void intervention_on_positive_result( model*, individual* );
void intervention_on_traced( model*, individual*, int, int, trace_token*, double, int );

void intervention_smart_release( model* );
int resolve_quarantine_reasons(int *);

#endif /* INTERVENTIONS_H_ */
