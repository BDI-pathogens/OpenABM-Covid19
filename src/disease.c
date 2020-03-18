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
	int time_event;
	infected->infector = infector;
	infected->time_infected = model->time;

	if( gsl_ran_bernoulli( rng, model->params->fraction_asymptomatic ) )
	{
		infected->status            = ASYMPTOMATIC;
		infected->time_asymptomatic = model->time;

		infected->next_disease_type = RECOVERED;
		time_event                  = model->time + sample_draw_list( model->asymptomatic_time_draws );
		infected->time_recovered    = time_event;
	}
	else
	{
		infected->status = PRESYMPTOMATIC;

		infected->next_disease_type = SYMPTOMATIC;
		time_event                  = model->time + sample_draw_list( model->symptomatic_time_draws );
		infected->time_symptomatic  = time_event;
	}

	infected->current_disease_event = add_individual_to_event_list( model, infected->status, infected, model->time );
	add_individual_to_event_list( model, infected->next_disease_type, infected, time_event);
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
void transition_to_symptomatic( model *model )
{
	long idx, n_infected;
	int time_event;
	double *hospitalised_fraction;
	event *event, *next_event;
	individual *indiv;

	n_infected 			  = model->event_lists[SYMPTOMATIC].n_daily_current[ model->time ];
	next_event 			  = model->event_lists[SYMPTOMATIC].events[ model->time ];
	hospitalised_fraction = model->params->hospitalised_fraction;

	for( idx = 0; idx < n_infected; idx++ )
	{
		event      = next_event;
		next_event = event->next;
		indiv      = event->individual;

		indiv->status = SYMPTOMATIC;
		remove_event_from_event_list( model, indiv->current_disease_event );

		if( gsl_ran_bernoulli( rng, hospitalised_fraction[ indiv->age_group ] ) )
		{
			indiv->next_disease_type = HOSPITALISED;
			time_event               = model->time + sample_draw_list( model->hospitalised_time_draws );
			indiv->time_hospitalised = time_event;
		}
		else
		{
			indiv->next_disease_type = RECOVERED;
			time_event               = model->time + sample_draw_list( model->recovered_time_draws );
			indiv->time_recovered    = time_event;
		}

		add_individual_to_event_list( model, indiv->next_disease_type, indiv, time_event );
		indiv->current_disease_event = event;

		if( indiv->quarantined == FALSE && gsl_ran_bernoulli( rng, model->params->self_quarantine_fraction ) )
		{
			set_quarantine_status( indiv, model->params, model->time, TRUE );
			indiv->quarantine_event = add_individual_to_event_list( model, QUARANTINED, indiv, model->time );
			add_individual_to_event_list( model, TEST_TAKE, indiv, model->time + 1 );
		}
	}
}

/*****************************************************************************************
*  Name:		transition_to_hospitalised
*  Description: Transitions symptomatic individual to hospital
*  Returns:		void
******************************************************************************************/
void transition_to_hospitalised( model *model )
{
	long idx, n_hospitalised;
	double time_event;
	double *fatality_rate;
	event *event, *next_event;
	individual *indiv;

	n_hospitalised = model->event_lists[HOSPITALISED].n_daily_current[ model->time ];
	next_event     = model->event_lists[HOSPITALISED].events[ model->time ];
	fatality_rate  = model->params->fatality_fraction;

	for( idx = 0; idx < n_hospitalised; idx++ )
	{
		event      = next_event;
		next_event = event->next;
		indiv      = event->individual;

		if( indiv->is_case == FALSE )
		{
			set_case( indiv, model->time );
			add_individual_to_event_list( model, CASE, indiv, model->time );
		}

		if( indiv->quarantined )
			release_individual_from_quarantine( model, event->individual );

		set_hospitalised( indiv, model->params, model->time );
		remove_event_from_event_list( model, indiv->current_disease_event );

		indiv->current_disease_event = event;
		if( gsl_ran_bernoulli( rng, fatality_rate[ indiv->age_group ] ) )
		{
			time_event               = model->time + sample_draw_list( model->death_time_draws );
			indiv->time_death        = time_event;
			indiv->next_disease_type = DEATH;
			add_individual_to_event_list( model, DEATH, indiv, time_event );
		}
		else
		{
			time_event               = model->time + sample_draw_list( model->recovered_time_draws );
			indiv->time_recovered    = time_event;
			indiv->next_disease_type = RECOVERED;
			add_individual_to_event_list( model, RECOVERED, indiv, time_event );
		};

		quarantine_contacts( model, indiv );
	}
}

/*****************************************************************************************
*  Name:		transition_to_recovered
*  Description: Transitions hospitalised and asymptomatic to recovered
*  Returns:		void
******************************************************************************************/
void transition_to_recovered( model *model )
{
	long idx, n_recovered;
	event *event, *next_event;
	individual *indiv;

	n_recovered = model->event_lists[RECOVERED].n_daily_current[ model->time ];
	next_event  = model->event_lists[RECOVERED].events[ model->time ];
	for( idx = 0; idx < n_recovered; idx++ )
	{
		event      = next_event;
		next_event = event->next;
		indiv      = event->individual;

		remove_event_from_event_list( model, indiv->current_disease_event );
		set_recovered( indiv, model->params, model->time );
	}
}

/*****************************************************************************************
*  Name:		transition_to_death
*  Description: Transitions hospitalised to death
*  Returns:		void
******************************************************************************************/
void transition_to_death( model *model )
{
	long idx, n_death;
	event *event, *next_event;
	individual *indiv;

	n_death    = model->event_lists[DEATH].n_daily_current[ model->time ];
	next_event = model->event_lists[DEATH].events[ model->time ];
	for( idx = 0; idx < n_death; idx++ )
	{
		event      = next_event;
		next_event = event->next;
		indiv      = event->individual;

		remove_event_from_event_list( model, indiv->current_disease_event );
		set_dead( indiv, model->time );
	}
}


