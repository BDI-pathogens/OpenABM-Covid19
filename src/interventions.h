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
struct trace_token_block{
	trace_token *trace_tokens;
	trace_token_block *next;
};

struct trace_token{
	individual *individual;
	individual *traced_from;
	trace_token *next_index;
	trace_token *last_index;
	trace_token *next;
	trace_token *last;
	int contact_time;
	int index_status;
};

struct vaccine{
	short idx;
	float *full_efficacy;				// efficacy against contracting the virus
	float *symptoms_efficacy;			// efficacy preventing symptoms
	float *severe_efficacy;			    // efficacy preventing severe symptoms (i.e. not needing to be hospitalised)
	short time_to_protect;				// time between having the vaccine and protection starting
	short vaccine_protection_period;	// time for which protections lasts
	short is_full;						// does it have some full protection
	short is_symptoms;					// does it have some symptoms-only protection
	short is_severe;					// does it have some severe-only protection
	short n_strains;					// the number of strains
	char name[INPUT_CHAR_LEN]; 			// unique name of the network
	vaccine *next;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void set_up_transition_times_intervention( model* );
void set_up_app_users( model* );
void set_up_risk_scores( model* );
void destroy_risk_scores( model* );
void update_intervention_policy( model*, int );

void set_up_trace_tokens( model*, float );
void add_trace_tokens( model*, float );
trace_token* create_trace_token( model*, individual*, int );
trace_token* index_trace_token( model*, individual* );
void remove_one_trace_token( model*, trace_token* );
void remove_traced_on_this_trace( model*, individual* );
void remove_traces_on_individual( model*, individual* );
void intervention_trace_token_release( model*, individual* );

int intervention_quarantine_until( model*, individual*, individual*, int, int, trace_token*, int, double );
void intervention_quarantine_release( model*, individual* );
void intervention_quarantine_household( model*, individual*, int, int, trace_token*, int );
void intervention_test_take( model*, individual* );
void intervention_test_result( model*, individual* );
void intervention_manual_trace( model *, individual *);
void intervention_notify_contacts( model*, individual*, int, trace_token*, int );
void intervention_index_case_symptoms_to_positive( model*, trace_token* );

short add_vaccine( model*, float*, float*, float*, short, short );
vaccine* get_vaccine_by_id( model*, short );
short intervention_vaccinate( model*, individual*, vaccine* );
short intervention_vaccinate_by_idx( model*, long, vaccine* );
long intervention_vaccinate_age_group( model*, double[ N_AGE_GROUPS ], vaccine*, long[ N_AGE_GROUPS ] );
void intervention_vaccine_protect( model*, individual*, void* );
void intervention_vaccine_wane( model*, individual*, void* );

void intervention_on_symptoms( model*, individual* );
void intervention_on_hospitalised( model*, individual* );
void intervention_on_critical( model*, individual* );
void intervention_on_positive_result( model*, individual* );
void intervention_on_traced( model*, individual*, int, int, trace_token*, double, int );

void intervention_smart_release( model* );
int resolve_quarantine_reasons(int *);

#endif /* INTERVENTIONS_H_ */
