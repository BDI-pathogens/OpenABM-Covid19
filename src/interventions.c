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
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

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
*  Name:		intervention_on_quarantine_until
*  Description: Quarantine an individual until a certain time
*  				If they are already in quarantine then extend quarantine until that time
*  Returns:		void
******************************************************************************************/
void intervention_quarantine_until( model *model, individual *indiv, int time, int maxof )
{
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
	}
	else
	{

		printf("%i %li\n", indiv->status, indiv->idx);
		print_exit( "releasing un-quarantined");
	}
}

/*****************************************************************************************
*  Name:		intervention_test_take
*  Description: An individual takes a test
*  Returns:		void
******************************************************************************************/
void intervention_test_take( model *model, individual *indiv )
{
	if( indiv->status == UNINFECTED || indiv->status == RECOVERED )
		indiv->quarantine_test_result = FALSE;
	else
		indiv->quarantine_test_result = TRUE;

	add_individual_to_event_list( model, TEST_RESULT, indiv, model->time + model->params->test_result_wait );
}

/*****************************************************************************************
*  Name:		intervention_test_result
*  Description: An individual gets a test result
*  Returns:		void
******************************************************************************************/
void intervention_test_result( model *model, individual *indiv )
{
	if( indiv->quarantine_test_result == FALSE && indiv->quarantined )
		intervention_quarantine_release( model, indiv );
	else
	{
		if( indiv->is_case == FALSE )
		{
			set_case( indiv, model->time );
			add_individual_to_event_list( model, CASE, indiv, model->time );
		}

		intervention_on_positive_result( model, indiv );
	}
}

/*****************************************************************************************
*  Name:		intervention_quarantine_contracts
*  Description: Quarantine contacts
*  Returns:		void
******************************************************************************************/
void intervention_quarantine_contacts( model *model, individual *indiv )
{
	interaction *inter;
	individual *contact;
	int idx, ddx, day, n_contacts, time_event, time_test;

	day = model->interaction_day_idx;
	for( ddx = 0; ddx < model->params->quarantine_days; ddx++ )
	{
		n_contacts = indiv->n_interactions[day];
		time_test  = model->time + max( model->params->test_insensititve_period - ddx, 1 );

		if( n_contacts > 0 )
		{
			inter = indiv->interactions[day];
			for( idx = 1; idx < n_contacts; idx++ )
			{
				if( inter->type != HOUSEHOLD )
					continue;

				contact = inter->individual;
				if( contact->status != HOSPITALISED && contact->status != DEATH && contact->quarantined == FALSE )
				{
					if( gsl_ran_bernoulli( rng, model->params->quarantine_fraction ) )
					{
						time_event = model->time + sample_transition_time( model, TRACED_QUARANTINE );

						if( model->params->quarantine_on_traced )
							intervention_quarantine_until( model, contact, time_event, TRUE );

						if( model->params->test_on_traced )
							add_individual_to_event_list( model, TEST_TAKE, contact, time_test );
					}
				}
				inter = inter->next;
			}
		}
		day = ifelse( day == 0, model->params->days_of_interactions -1, day-1 );
	}
}

/*****************************************************************************************
*  Name:		intervention_on_symptoms
*  Description: The interventions performed upon showing symptoms of a flu-like
*  				illness
*  Returns:		void
******************************************************************************************/
void intervention_on_symptoms( model *model, individual *indiv )
{
	if( indiv->quarantined == FALSE && gsl_ran_bernoulli( rng, model->params->self_quarantine_fraction ) )
	{
		int time_event = model->time + sample_transition_time( model, SYMPTOMATIC_QUARANTINE );
		intervention_quarantine_until( model, indiv, time_event, TRUE );

		if( model->params->test_on_symptoms )
			add_individual_to_event_list( model, TEST_TAKE, indiv, model->time + 1 );
	}
}

/*****************************************************************************************
*  Name:		intervention_on_hospitalised
*  Description: The interventions performed upon becoming hopsitalised.
*  					1. Make clinical diagnosis of case without testing
*  					2. Quarantine contacts
*  Returns:		void
******************************************************************************************/
void intervention_on_hospitalised( model *model, individual *indiv )
{
	if( indiv->is_case == FALSE )
	{
		set_case( indiv, model->time );
		add_individual_to_event_list( model, CASE, indiv, model->time );
	}

	if( model->params->quarantine_on_traced || model->params->test_on_traced )
		intervention_quarantine_contacts( model, indiv );
}

/*****************************************************************************************
*  Name:		intervention_on_positive_result
*  Description: The interventions performed upon receiving a positive test result
*  Returns:		void
******************************************************************************************/
void intervention_on_positive_result( model *model, individual *indiv )
{

	if( indiv->status != HOSPITALISED )
	{
		int time_event = model->time + sample_transition_time( model, TEST_RESULT_QUARANTINE );
		intervention_quarantine_until( model, indiv, time_event, TRUE );

		if( model->params->quarantine_on_traced || model->params->test_on_traced )
			intervention_quarantine_contacts( model, indiv );
	}
}



