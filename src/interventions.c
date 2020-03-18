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
*  Name:		intervention_on_quarantine_release
*  Description: Release an individual held in quarantine
*  Returns:		void
******************************************************************************************/
void intervention_quarantine_release( model *model, individual *indiv )
{
	remove_event_from_event_list( model, indiv->quarantine_event );
	if( indiv->quarantine_release_event != NULL )
		remove_event_from_event_list( model, indiv->quarantine_release_event );
	set_quarantine_status( indiv, model->params, model->time, FALSE );
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
	if( indiv->quarantine_test_result == FALSE )
		indiv->quarantine_release_event = add_individual_to_event_list( model, QUARANTINE_RELEASE, indiv, model->time );
	else
	{
		if( indiv->is_case == FALSE )
		{
			set_case( indiv, model->time );
			add_individual_to_event_list( model, CASE, indiv, model->time );
		}

		if( indiv->status != HOSPITALISED )
		{
			indiv->quarantine_release_event = add_individual_to_event_list( model, QUARANTINE_RELEASE, indiv, model->time + 14 );
			intervention_quarantine_contacts( model, indiv );
		}
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
	int idx, ddx, day, n_contacts;
	int time_event;

	day = model->interaction_day_idx;
	for( ddx = 0; ddx < model->params->quarantine_days; ddx++ )
	{
		n_contacts = indiv->n_interactions[day];
		time_event = model->time + max( model->params->test_insensititve_period - ddx, 1 );

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
						set_quarantine_status( contact, model->params, model->time, TRUE );
						contact->quarantine_event = add_individual_to_event_list( model, QUARANTINED, contact, model->time );
						add_individual_to_event_list( model, TEST_TAKE, contact, time_event );
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
		set_quarantine_status( indiv, model->params, model->time, TRUE );
		indiv->quarantine_event = add_individual_to_event_list( model, QUARANTINED, indiv, model->time );
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

	intervention_quarantine_contacts( model, indiv );
}



