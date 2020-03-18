/*
 * disease.c


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
#include "structure.h"
#include "interventions.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

/*****************************************************************************************
*  Name:		transmit_virus_by_type
*  Description: Transmits virus over the interaction network for a type of
*  				infected people
*  Returns:		void
******************************************************************************************/
void transmit_virus_by_type(
	model *model,
	int type
)
{
	long idx, jdx, n_infected;
	int day, n_interaction;
	double hazard_rate;
	event_list *list = &(model->event_lists[type]);
	event *event, *next_event;
	interaction *interaction;
	individual *infector;

	for( day = model->time-1; day >= max( 0, model->time - MAX_INFECTIOUS_PERIOD ); day-- )
	{
		hazard_rate = list->infectious_curve[ model->time- 1 - day ];
		n_infected  = list->n_daily_current[ day];
		next_event  = list->events[ day ];

		for( idx = 0; idx < n_infected; idx++ )
		{
			event      = next_event;
			next_event = event->next;

			infector      = event->individual;
			n_interaction = infector->n_interactions[ model->interaction_day_idx ];
			interaction   = infector->interactions[ model->interaction_day_idx ];

			for( jdx = 0; jdx < n_interaction; jdx++ )
			{
				if( interaction->individual->status == UNINFECTED )
				{
					interaction->individual->hazard -= hazard_rate;
					if( interaction->individual->hazard < 0 )
						new_infection( model, interaction->individual, infector );
				}
				interaction = interaction->next;
			}
		}
	}
}

/*****************************************************************************************
*  Name:		transmit_virus
*  Description: Transmits virus over the interaction network
*
*  				Transmission by groups depending upon disease status.
*  				Note that quarantine is not a disease status and they will either
*  				be presymptomatic/symptomatic/asymptomatic and quarantining is
*  				modelled by reducing the number of interactions in the network.
*
*  Returns:		void
******************************************************************************************/
void transmit_virus( model *model )
{
	transmit_virus_by_type( model, PRESYMPTOMATIC );
	transmit_virus_by_type( model, SYMPTOMATIC );
	transmit_virus_by_type( model, ASYMPTOMATIC );
	transmit_virus_by_type( model, HOSPITALISED );
}

/*****************************************************************************************
*  Name:		new_infection
*  Description: infects a new individual
*  Returns:		void
******************************************************************************************/
void new_infection(
	model *model,
	individual *infected,
	individual *infector
)
{
	infected->infector = infector;

	if( gsl_ran_bernoulli( rng, model->params->fraction_asymptomatic ) )
	{
		transition_one_disese_event( model, infected, NO_EVENT, ASYMPTOMATIC, NO_EDGE );
		transition_one_disese_event( model, infected, ASYMPTOMATIC, RECOVERED, ASYMPTOMATIC_RECOVERED );
	}
	else
	{
		transition_one_disese_event( model, infected, NO_EVENT, PRESYMPTOMATIC, NO_EDGE );
		transition_one_disese_event( model, infected, PRESYMPTOMATIC, SYMPTOMATIC, PRESYMPTOMATIC_SYMPTOMATIC );
	}
}

/*****************************************************************************************
*  Name:		transition_one_disease_event
*  Description: Generic function for updating an individual with their next
*				event and adding the applicable events
*  Returns:		void
******************************************************************************************/
void transition_one_disese_event(
	model *model,
	individual *indiv,
	int from,
	int to,
	int edge
)
{
	indiv->status           = from;
	indiv->time_event[from] = model->time;

	if( indiv->current_disease_event != NULL )
		remove_event_from_event_list( model, indiv->current_disease_event );
	if( indiv->next_disease_event != NULL )
		indiv->current_disease_event = indiv->next_disease_event;

	if( to != NO_EVENT )
	{
		indiv->time_event[to]     = model->time + ifelse( edge == NO_EDGE, 0, sample_transition_time( model, edge ) );
		indiv->next_disease_event = add_individual_to_event_list( model, to, indiv, indiv->time_event[to] );
	}
}

/*****************************************************************************************
*  Name:		transition_to_symptomatic
*  Description: Transitions infected who are due to become symptomatic. At this point
*  				there are 2 choices to be made:
*
*  				1. Will the individual require hospital treatment or will
*  				they recover without needing treatment
*  				2. Does the individual self-quarantine at this point and
*  				asks for a test
*
*  Returns:		void
******************************************************************************************/
void transition_to_symptomatic( model *model, individual *indiv )
{
	if( gsl_ran_bernoulli( rng, model->params->hospitalised_fraction[ indiv->age_group ] ) )
		transition_one_disese_event( model, indiv, SYMPTOMATIC, HOSPITALISED, SYMPTOMATIC_HOSPITALISED );
	else
		transition_one_disese_event( model, indiv, SYMPTOMATIC, RECOVERED, SYMPTOMATIC_RECOVERED );

	if( indiv->quarantined == FALSE && gsl_ran_bernoulli( rng, model->params->self_quarantine_fraction ) )
	{
		set_quarantine_status( indiv, model->params, model->time, TRUE );
		indiv->quarantine_event = add_individual_to_event_list( model, QUARANTINED, indiv, model->time );
		add_individual_to_event_list( model, TEST_TAKE, indiv, model->time + 1 );
	}
}

/*****************************************************************************************
*  Name:		transition_to_hospitalised
*  Description: Transitions symptomatic individual to hospital
*  Returns:		void
******************************************************************************************/
void transition_to_hospitalised( model *model, individual *indiv )
{
	if( indiv->is_case == FALSE )
	{
		set_case( indiv, model->time );
		add_individual_to_event_list( model, CASE, indiv, model->time );
	}

	set_hospitalised( indiv, model->params, model->time );

	if( gsl_ran_bernoulli( rng, model->params->fatality_fraction[ indiv->age_group ] ) )
		transition_one_disese_event( model, indiv, HOSPITALISED, DEATH, HOSPITALISED_DEATH );
	else
		transition_one_disese_event( model, indiv, HOSPITALISED, RECOVERED, HOSPITALISED_RECOVERED );

	if( indiv->quarantined )
		intervention_quarantine_release( model, indiv );

	quarantine_contacts( model, indiv );
}

/*****************************************************************************************
*  Name:		transition_to_recovered
*  Description: Transitions hospitalised and asymptomatic to recovered
*  Returns:		void
******************************************************************************************/
void transition_to_recovered( model *model, individual *indiv )
{
	transition_one_disese_event( model, indiv, HOSPITALISED, NO_EVENT, NO_EDGE );
	set_recovered( indiv, model->params, model->time );
}

/*****************************************************************************************
*  Name:		transition_to_death
*  Description: Transitions hospitalised to death
*  Returns:		void
******************************************************************************************/
void transition_to_death( model *model, individual *indiv )
{
	transition_one_disese_event( model, indiv, DEATH, NO_EVENT, NO_EDGE );
	set_dead( indiv, model->time );
}


