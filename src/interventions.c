/*
 * interventions.c
 *
 *  Created on: 18 Mar 2020
 *      Author: hinchr
 */

#include "model.h"
#include "individual.h"
#include "utilities.h"
#include "constant.h"
#include "params.h"
#include "network.h"
#include "disease.h"
#include "interventions.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*****************************************************************************************
*  Name:		set_up_transition_times
*  Description: sets up discrete distributions for the times it takes to
*  				transition along edges of the intervention
*
*  Returns:		void
******************************************************************************************/
void set_up_transition_times_intervention( model *model )
{
	parameters *params = model->params;
	int **transitions  = model->transition_time_distributions;

	geometric_max_draw_list( transitions[SYMPTOMATIC_QUARANTINE], N_DRAW_LIST, params->quarantine_dropout_self,     params->quarantine_length_self );
	geometric_max_draw_list( transitions[TRACED_QUARANTINE],      N_DRAW_LIST, params->quarantine_dropout_traced,   params->quarantine_length_traced );
	geometric_max_draw_list( transitions[TEST_RESULT_QUARANTINE], N_DRAW_LIST, params->quarantine_dropout_positive, params->quarantine_length_positive );
}

/*****************************************************************************************
*  Name:		set_up_app_users
*  Description: Set up the proportion of app users in the population (default is FALSE)
******************************************************************************************/
void set_up_app_users( model *model )
{
	long idx, jdx, age, current_users, not_users, max_user;
	double *fraction = model->params->app_users_fraction;

	for( age = 0; age < N_AGE_GROUPS; age++ )
	{
		current_users = 0;
		not_users     = 0;
		for( idx = 0; idx < model->params->n_total; idx++ )
			if( model->population[ idx ].age_group == age )
			{
				current_users += model->population[ idx ].app_user;
				not_users     += 1 - model->population[ idx ].app_user;
			}

		max_user = ceil( ( current_users + not_users ) * fraction[age] ) - current_users;
		if( max_user < 0 || max_user > not_users )
			print_exit( "Bad target app_fraction_users" );

		int *users = calloc( not_users, sizeof( int ) );

		for( idx = 0; idx < max_user; idx++ )
			users[ idx ] = 1;

		gsl_ran_shuffle( rng, users, not_users, sizeof( int ) );

		jdx   = 0;
		for( idx = 0; idx < model->params->n_total; idx++ )
			if( model->population[ idx ].age_group == age && model->population[ idx ].app_user == FALSE )
				model->population[ idx ].app_user = users[ jdx++ ];

		free( users );
	}
};

/*****************************************************************************************
*  Name:		set_up_trace_tokens
*  Description: sets up the stock trace_tokens note that these get recycled once we
*  				move to a later date
*  Returns:		void
******************************************************************************************/
void set_up_trace_tokens( model *model )
{
	double tokens_per_person = 3;
	model->n_trace_tokens = ceil(  model->params->n_total * tokens_per_person );
	long idx;

	model->trace_tokens = calloc( model->n_trace_tokens, sizeof( trace_token ) );

	model->trace_tokens[0].next_index = NULL;
	for( idx = 1; idx < model->n_trace_tokens; idx++ )
		model->trace_tokens[idx].next_index = &(model->trace_tokens[idx-1]);

	model->next_trace_token = &(model->trace_tokens[ model->n_trace_tokens - 1 ]);
	model->n_trace_tokens_used = 0;
}

/*****************************************************************************************
*  Name:		new_trace_token
*  Description: gets a new trace token
*  Returns:		void
******************************************************************************************/
trace_token* new_trace_token( model *model, individual *indiv, int contact_time )
{
	trace_token *token = model->next_trace_token;

	model->next_trace_token = token->next_index;

	token->last = NULL;
	token->next = NULL;
	token->next_index = NULL;
	token->individual = indiv;
	token->days_since_contact = model->time - contact_time;
	model->n_trace_tokens_used++;

	if( model->n_trace_tokens == model->n_trace_tokens_used)
		print_exit( "run out of trace tokens");

	return token;
}

/*****************************************************************************************
*  Name:		index_trace_token
*  Description: get the index trace token at the start of a tracing cascade and
*  				assigns it to the indiviual, note if the individual already has
*  				one then we just add it to it
*  Returns:		void
******************************************************************************************/
trace_token* index_trace_token( model *model, individual *indiv )
{
	if( indiv->index_trace_token == NULL )
		indiv->index_trace_token = new_trace_token( model, indiv, model->time );

	return indiv->index_trace_token;
}

/*****************************************************************************************
*  Name:		remove_traced_on_this_trace
*  Description: add the end of a tracing event this removes the tag which
*  				prevents us from double counting people
*  Returns:		void
******************************************************************************************/
void remove_traced_on_this_trace( model *model, individual *indiv )
{
	trace_token *token = indiv->index_trace_token;

	while( token->next_index != NULL )
	{
		token = token->next_index;
		token->individual->traced_on_this_trace = FALSE;
	}
	indiv->traced_on_this_trace = FALSE;

}

/*****************************************************************************************
*  Name:		update_intervention_policy
*  Description: Updates the intervention policy by adjusting parmaters
******************************************************************************************/
void update_intervention_policy( model *model, int time )
{
	parameters *params = model->params;
	int type;

	if( time == 0 )
	{
		params->app_turned_on       = FALSE;
		params->lockdown_on	        = FALSE;
		params->lockdown_elderly_on	= FALSE;
		for( type = 0; type < N_WORK_NETWORKS; type++ )
			params->daily_fraction_work_used[type] = params->daily_fraction_work;

		for( type = 0; type < N_INTERACTION_TYPES; type++ )
			params->relative_transmission_used[type] = params->relative_transmission[type];

		params->interventions_on = ( params->intervention_start_time == 0 );
	}

	if( time == params->intervention_start_time )
		params->interventions_on = TRUE;

	if( time == params->app_turn_on_time )
		set_model_param_app_turned_on( model, TRUE );

	if( time == params->lockdown_time_on )
		set_model_param_lockdown_on( model, TRUE );

	if( time == params->lockdown_time_off )
		set_model_param_lockdown_on( model, FALSE );
	
	if( time == params->lockdown_elderly_time_on )
		set_model_param_lockdown_elderly_on( model, TRUE );

	if( time == params->lockdown_elderly_time_off )
		set_model_param_lockdown_elderly_on( model, FALSE );

	if( time == params->testing_symptoms_time_on )
		set_model_param_test_on_symptoms( model, TRUE );

	if( time == params->testing_symptoms_time_off )
		set_model_param_test_on_symptoms( model, FALSE );
};

/*****************************************************************************************
*  Name:		intervention_on_quarantine_until
*  Description: Quarantine an individual until a certain time
*  				If they are already in quarantine then extend quarantine until that time
*  Returns:		void
******************************************************************************************/
void intervention_quarantine_until(
	model *model,
	individual *indiv,
	int time,
	int maxof,
	trace_token *index_token,
	int contact_time
)
{
	if( indiv->traced_on_this_trace )
		return;

	if( is_in_hospital( indiv) )
		return;

	if( index_token != NULL )
	{
		// add the trace token to their list
		trace_token *token = new_trace_token( model, indiv, contact_time );

		if( indiv->trace_tokens != NULL )
		{
			token->next = indiv->trace_tokens;
			indiv->trace_tokens->last = token;
		}
		indiv->trace_tokens = token;
		indiv->traced_on_this_trace = TRUE;

		// then add it to the index_token list
		token->next_index = index_token->next_index;
		index_token->next_index = token;
	}

	if( time == model->time )
		return;

	if( indiv->quarantine_event == NULL )
	{
		indiv->quarantine_event = add_individual_to_event_list( model, QUARANTINED, indiv, model->time );
		set_quarantine_status( indiv, model->params, model->time, TRUE );
	}

	if( indiv->quarantine_release_event != NULL )
	{
		if( maxof && indiv->quarantine_release_event->time > time )
			return;

		remove_event_from_event_list( model, indiv->quarantine_release_event );
	}

	indiv->quarantine_release_event = add_individual_to_event_list( model, QUARANTINE_RELEASE, indiv, time );
}

/*****************************************************************************************
*  Name:		intervention_on_quarantine_release
*  Description: Release an individual held in quarantine
*  Returns:		void
******************************************************************************************/
void intervention_quarantine_release( model *model, individual *indiv )
{
	if( indiv->quarantine_release_event != NULL )
		remove_event_from_event_list( model, indiv->quarantine_release_event );

	if( indiv->quarantine_event != NULL )
	{
		remove_event_from_event_list( model, indiv->quarantine_event );
		set_quarantine_status( indiv, model->params, model->time, FALSE );
	};
}

/*****************************************************************************************
*  Name:		intervention_test_order
*  Description: Order a test for either today or a future date
*  Returns:		void
******************************************************************************************/
void intervention_test_order( model *model, individual *indiv, int time )
{
	if( indiv->quarantine_test_result == NO_TEST && !(indiv->is_case) )
	{
		add_individual_to_event_list( model, TEST_TAKE, indiv, time );
		indiv->quarantine_test_result = TEST_ORDERED;
	}
}

/*****************************************************************************************
*  Name:		intervention_test_take
*  Description: An individual takes a test
*
*  				At the time of testing it will test positive if the individual has
*  				had the virus for less time than it takes for the test to be sensitive
*  Returns:		void
******************************************************************************************/
void intervention_test_take( model *model, individual *indiv )
{
	if( indiv->status == SUSCEPTIBLE || indiv->status == RECOVERED )
		indiv->quarantine_test_result = FALSE;
	else
	{
		if( model->time - time_infected( indiv ) >= model->params->test_insensitive_period )
			indiv->quarantine_test_result = TRUE;
		else
			indiv->quarantine_test_result = FALSE;
	}

	add_individual_to_event_list( model, TEST_RESULT, indiv, model->time + model->params->test_result_wait );
}

/*****************************************************************************************
*  Name:		intervention_test_result
*  Description: An individual gets a test result
*
*  				 1. On a negative result the person is released from quarantine
*  				 2. On a positive result they become a case and trigger the
*  				 	intervention_on_positive_result cascade
*  Returns:		void
******************************************************************************************/
void intervention_test_result( model *model, individual *indiv )
{
	if( indiv->quarantine_test_result == FALSE )
	{
		if( indiv->quarantined )
			intervention_quarantine_release( model, indiv );

		intervention_trace_token_release( model, indiv );
	}
	else
	{
		if( indiv->is_case == FALSE )
		{
			set_case( indiv, model->time );
			add_individual_to_event_list( model, CASE, indiv, model->time );
		}

		if( !is_in_hospital( indiv ) || !(model->params->allow_clinical_diagnosis) )
			intervention_on_positive_result( model, indiv );
	}
	indiv->quarantine_test_result = NO_TEST;
}

/*****************************************************************************************
*  Name:		intervention_notify_contracts
*  Description: If the individual is an app user then we loop over the stored contacts
*  				and notifies them.
*  				Start from the oldest date, so that if we have met a contact
*  				multiple times we order the test from the first contact
*  Returns:		void
******************************************************************************************/
void intervention_notify_contacts(
	model *model,
	individual *indiv,
	int recursion_level,
	trace_token *index_token
)
{
	if( !indiv->app_user || !model->params->app_turned_on )
		return;

	interaction *inter;
	individual *contact;
	parameters *params = model->params;
	int idx, ddx, day, n_contacts;

	day = model->interaction_day_idx;

	for( ddx = 0; ddx < params->quarantine_days; ddx++ )
	{
		n_contacts = indiv->n_interactions[day];

		if( n_contacts > 0 )
		{
			inter = indiv->interactions[day];
			for( idx = 0; idx < n_contacts; idx++ )
			{
				contact = inter->individual;
				if( contact->app_user )
				{
					if( inter->traceable == UNKNOWN )
						inter->traceable = gsl_ran_bernoulli( rng, params->traceable_interaction_fraction );
					if( inter->traceable )
						intervention_on_traced( model, contact, model->time - ddx, recursion_level, index_token );
				}
				inter = inter->next;
			}
		}
		ring_dec( day, model->params->days_of_interactions );
	}
}

/*****************************************************************************************
*  Name:		intervention_trace_token_release
*  Description: what to do when a trace_token is released
*  Returns:		void
******************************************************************************************/
void intervention_trace_token_release( model *model, individual *indiv )
{
	individual *contact;
	trace_token *token = indiv->index_trace_token;
	trace_token *next_token;
	int zero_traced = FALSE;

	if( token == NULL )
		return;

	// if nobody traced then nothing to do
	if( token->next_index == NULL )
		zero_traced = TRUE;
	else
		next_token = token->next_index;

	if( zero_traced & ( (token->next != NULL) | (token->last != NULL) ) )
		print_exit( "error index token should not link to a list");

	// return the index token to the stack
	token->next_index = model->next_trace_token;
	model->next_trace_token = token;
	model->n_trace_tokens_used--;
	indiv->index_trace_token = NULL;

	if( zero_traced )
		return;

	while( next_token != NULL )
	{
		// get the next token on the list of this person trace_token
		token = next_token;
		next_token = token->next_index;

		// remove the token from the individual and return it to the stack
		contact = token->individual;
		if( contact->trace_tokens == token )
		{
			contact->trace_tokens = token->next;
			if( contact->trace_tokens != NULL )
				contact->trace_tokens->last = NULL;
			else
			if( contact->index_trace_token == NULL )
				intervention_quarantine_release( model, contact );
		}
		else
		{
			token->last->next = token->next;
			if( token->next != NULL )
				token->next->last = token->last;
		}

		// put the token back on the stack
		token->next_index = model->next_trace_token;
		model->next_trace_token = token;
		model->n_trace_tokens_used--;
	}
}

/*****************************************************************************************
*  Name:		intervention_quarantine_household
*  Description: Quarantine everyone in a household
*  Returns:		void
******************************************************************************************/
void intervention_quarantine_household(
	model *model,
	individual *indiv,
	int time,
	int contact_trace,
	trace_token *index_token,
	int contact_time
)
{
	individual *contact;
	int idx, n, time_event;
	long* members;

	n          = model->household_directory->n_jdx[indiv->house_no];
	members    = model->household_directory->val[indiv->house_no];
	time_event = ifelse( time != UNKNOWN, time, model->time + sample_transition_time( model, TRACED_QUARANTINE ) );

	for( idx = 0; idx < n; idx++ )
		if( members[idx] != indiv->idx )
		{
			contact = &(model->population[members[idx]]);

			if( contact->status == DEATH || is_in_hospital( contact ) )
				continue;

			intervention_quarantine_until( model, contact, time_event, TRUE, index_token, contact_time );

			if( contact_trace && ( model->params->quarantine_on_traced || model->params->test_on_traced ) )
				intervention_notify_contacts( model, contact, NOT_RECURSIVE, index_token );
		}
}

/*****************************************************************************************
*  Name:		intervention_on_symptoms
*  Description: The interventions performed upon showing symptoms of a flu-like symptoms
*
*  				 1. If in quarantine already or drawn for self-quarantine then
*  				    quarantine for length of time symptomatic people do
*  				 2. If we have community testing on symptoms and a test has not been
*  				    ordered already then order one
*  				 3. Option to quarantine all household members upon symptoms
*  Returns:		void
******************************************************************************************/
void intervention_on_symptoms( model *model, individual *indiv )
{
	if( !model->params->interventions_on )
		return;

	int quarantine, time_event;
	parameters *params = model->params;

	quarantine = indiv->quarantined || gsl_ran_bernoulli( rng, params->self_quarantine_fraction );

	if( quarantine )
	{
		trace_token *index_token = index_trace_token( model, indiv );

		time_event = model->time + sample_transition_time( model, SYMPTOMATIC_QUARANTINE );
		intervention_quarantine_until( model, indiv, time_event, TRUE, NULL, model->time );
		indiv->traced_on_this_trace = TRUE;

		if( params->quarantine_household_on_symptoms )
			intervention_quarantine_household( model, indiv, time_event, params->quarantine_household_contacts_on_symptoms, index_token, model->time );

		if( params->test_on_symptoms )
			intervention_test_order( model, indiv, model->time + params->test_order_wait );

		if( params->trace_on_symptoms && ( params->quarantine_on_traced || params->test_on_traced ) )
			intervention_notify_contacts( model, indiv, 1, index_token );

		remove_traced_on_this_trace( model, indiv );
		add_individual_to_event_list( model, TRACE_TOKEN_RELEASE, indiv, model->time + params->quarantine_length_traced );
	}
}

/*****************************************************************************************
*  Name:		intervention_on_hospitalised
*  Description: The interventions performed upon becoming hopsitalised.
*
*  					1. Take a test immediately
*  					2. Option to use a clinical diagnosis to trigger contact tracing
*
*  Returns:		void
******************************************************************************************/
void intervention_on_hospitalised( model *model, individual *indiv )
{
	if( !model->params->interventions_on )
		return;

	intervention_test_order( model, indiv, model->time );

	if( model->params->allow_clinical_diagnosis )
		intervention_on_positive_result( model, indiv );
}

/*****************************************************************************************
*  Name:		intervention_on_positive_result
*  Description: The interventions performed upon receiving a positive test result
*
*  				 1. Patients who are not in hospital will be quarantined
*  				 2. Commence contact-tracing for patients not in hospital or hospital
*  				    patients if clinical diagnosis not being used as a trigger
*  Returns:		void
******************************************************************************************/
void intervention_on_positive_result( model *model, individual *indiv )
{
	if( !model->params->interventions_on )
		return;

	int time_event = UNKNOWN;
	parameters *params = model->params;
	trace_token *index_token = index_trace_token( model, indiv );

	if( !is_in_hospital( indiv ) )
	{
		time_event = model->time + sample_transition_time( model, TEST_RESULT_QUARANTINE );
		intervention_quarantine_until( model, indiv, time_event, TRUE, NULL, model->time );
	}
	indiv->traced_on_this_trace = TRUE;

	if( params->quarantine_household_on_positive )
		intervention_quarantine_household( model, indiv, time_event, params->quarantine_household_contacts_on_positive, index_token, model->time );

	if( params->trace_on_positive && ( params->quarantine_on_traced || params->test_on_traced ) )
		intervention_notify_contacts( model, indiv, 1, index_token );

	remove_traced_on_this_trace( model, indiv );
	add_individual_to_event_list( model, TRACE_TOKEN_RELEASE, indiv, model->time + params->quarantine_length_traced );
}

/*****************************************************************************************
*  Name:		intervention_on_critical
*  Description: The interventions performed upon becoming critical
*  Returns:		void
******************************************************************************************/
void intervention_on_critical( model *model, individual *indiv )
{
	if( !model->params->interventions_on )
		return;
}

/*****************************************************************************************
*  Name:		intervention_on_traced
*  Description: Optional interventions performed upon becoming contact-traced
*
*   			1. Quarantine the individual
*   			2. Quarantine the individual and their household
*  				2. Order a test for the individual
*  				4. Recursive contact-trace
*
*  Arguments:	model 		 	- pointer to model
*  				indiv 		 	- pointer to person being traced
*  				contact_time 	- time at which the contact was made
*				recursion level - layers of the network to reach this connection
*
*  Returns:		void
******************************************************************************************/
void intervention_on_traced(
	model *model,
	individual *indiv,
	int contact_time,
	int recursion_level,
	trace_token *index_token
)
{
	if( is_in_hospital( indiv ) || indiv->is_case )
		return;

	parameters *params = model->params;

	if( params->quarantine_on_traced )
	{
		int time_event = model->time + sample_transition_time( model, TRACED_QUARANTINE );
		intervention_quarantine_until( model, indiv, time_event, TRUE, index_token, contact_time );

		if( params->quarantine_household_on_traced && recursion_level != NOT_RECURSIVE )
			intervention_quarantine_household( model, indiv, time_event, FALSE, index_token, contact_time );
	}

	if( params->test_on_traced )
	{
		int time_test = max( model->time + params->test_order_wait, contact_time + params->test_insensitive_period );
		intervention_test_order( model, indiv, time_test );
	}

	if( recursion_level != NOT_RECURSIVE && recursion_level < params->tracing_network_depth )
		intervention_notify_contacts( model, indiv, recursion_level + 1, index_token );
}

/*****************************************************************************************
*  Name:		intervention_smart_release
*  Description: Release people from quarantine based upon how many people have developed
*				symptoms following being quarantined by the index case
*  Returns:		void
******************************************************************************************/
void intervention_smart_release( model *model )
{
	long idx, n_events;
	int day,n_symptoms, time_index;
	individual *indiv, *contact;
	event *event, *next_event;
	trace_token *token;

	int days =  model->params->quarantine_smart_release_day;

	day        = model->time + model->params->quarantine_length_traced - days;
	time_index = model->time - days;

	if( time_index < 1)
		return;

	n_events    = model->event_lists[TRACE_TOKEN_RELEASE].n_daily_current[  day ];
	next_event  = model->event_lists[TRACE_TOKEN_RELEASE].events[day ];

	for( idx = 0; idx < n_events; idx++ )
	{
		event      = next_event;
		next_event = event->next;
		indiv      = event->individual;

		n_symptoms = 0;

		token = indiv->index_trace_token;
		if( token == NULL )
			continue;

		if( is_in_hospital( indiv ) || ( indiv->time_event[CASE] >= time_index) )
			continue;

		token = token->next_index;
		while( token != NULL )
		{
			contact = token->individual;
			if( contact->time_event[SYMPTOMATIC] >= time_index )
			{
				n_symptoms++;
				break;
			}
			token = token->next_index;
		}

		if( n_symptoms == 0 )
		{
			intervention_trace_token_release( model, indiv );
		}
	}
}

