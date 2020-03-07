/*
 * model.c
 *
 *  Created on: 5 Mar 2020
 *      Author: hinchr
 */

#include "model.h"
#include "individual.h"
#include "utilities.h"
#include "constant.h"
#include "params.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

/*****************************************************************************************
*  Name:		new_model
*  Description: Builds a new model object from a parameters object and returns a
*  				pointer to it.
*  				 1. Creates memory for it
*  				 2. Initialises the gsl random numbers generator
*  Returns:		pointer to model
******************************************************************************************/
model* new_model( parameters *params )
{
	model *model  = calloc( 1, sizeof( model ) );
	model->params = *params;
	model->time   = 0;

	set_up_population( model );
	set_up_interactions( model );
	set_up_events( model );
	set_up_distributions( model );
	set_up_seed_infection( model );

	return model;
};

/*****************************************************************************************
*  Name:		destroy_model
*  Description: Destroys the model structure and releases its memory
******************************************************************************************/
void destroy_model( model *model )
{
	parameters *params = &(model->params);
	long idx;

	for( idx = 0; idx < params->n_total; idx++ )
		destroy_individual( &(model->population[idx] ) );

	free( model->population );
    free( model->possible_interactions );
    free( model->interactions );
    free( model );
};

/*****************************************************************************************
*  Name:		set_up_events
*  Description: sets up the event tags
*  Returns:		void
******************************************************************************************/
void set_up_events( model *model )
{
	parameters *params = &(model->params);
	int types = 3;

	model->event_idx = 0;
	model->events    = calloc( types * params->n_total, sizeof( event ) );
}

/*****************************************************************************************
*  Name:		set_up_population
*  Description: sets up the initial population
*  Returns:		void
******************************************************************************************/
void set_up_population( model *model )
{
	parameters *params = &(model->params);
	long idx;

	model->population = calloc( params->n_total, sizeof( individual ) );
	for( idx = 0; idx < params->n_total; idx++ )
		initialize_individual( &(model->population[idx]), params, idx );
}

/*****************************************************************************************
*  Name:		set_up_interactions
*  Description: sets up the stock of interactions, note that these get recycled once we
*  				move to a later date
*  Returns:		void
******************************************************************************************/
void set_up_interactions( model *model )
{
	parameters *params = &(model->params);
	individual *indiv;
	long idx, n_idx, indiv_idx;

	// FIXME - need to a good estimate of the total number of interactions
	//         easy at the moment since we have a fixed number per individual
	long n_daily_interactions = params->n_total * params->mean_daily_interactions;
	long n_interactions       = n_daily_interactions * params->days_of_interactions;

	model->interactions          = calloc( n_interactions, sizeof( interaction ) );
	model->n_interactions        = n_interactions;
	model->interaction_idx       = 0;
	model->interaction_day_idx   = 0;

	model->possible_interactions = calloc( n_daily_interactions, sizeof( long ) );
	idx = 0;
	for( indiv_idx = 0; indiv_idx < params->n_total; indiv_idx++ )
	{
		indiv = &(model->population[ indiv_idx ]);
		for( n_idx = 0; n_idx < indiv->n_mean_interactions; n_idx++ )
			model->possible_interactions[ idx++ ] = indiv_idx;
	}

	model->n_possible_interactions = idx;
}

/*****************************************************************************************
*  Name:		set_up_distributions
*  Description: sets up discrete distributions and functions which are used to
*  				model events
*  Returns:		void
******************************************************************************************/
void set_up_distributions( model *model )
{
	parameters *params = &(model->params);

	gamma_draw_list( model->symptomatic_draws, N_DRAW_LIST, params->mean_time_to_symptoms, params->sd_time_to_symptoms );

	gamma_rate_curve( model->infected.infectious_curve, MAX_INFECTIOUS_PERIOD, params->mean_infectious_period,
					  params->sd_infectious_period, params->infectious_rate / params->mean_daily_interactions );

	gamma_rate_curve( model->symptomatic.infectious_curve, MAX_INFECTIOUS_PERIOD, params->mean_infectious_period,
				      params->sd_infectious_period, params->infectious_rate/ params->mean_daily_interactions );
}

/*****************************************************************************************
*  Name:		new_event
*  Description: gets a new event tag
*  Returns:		void
******************************************************************************************/
event* new_event( model *model )
{
	return &(model->events[ model->event_idx++ ] );
}

/*****************************************************************************************
*  Name:		transmit_virus_by_type
*  Description: Transmits virus over the interaction network for a type of
*  				infected people
*  Returns:		void
******************************************************************************************/
void transmit_virus_by_type(
	model *model,
	event_list *list
)
{
	long idx, jdx, n_infected, tot;
	int day, n_interaction;
	double hazard_rate;
	event *event;
	interaction *interaction;
	individual *infector;

	tot = 0;
	for( day = model->time-1; day >= max( 0, model->time - MAX_INFECTIOUS_PERIOD ); day-- )
	{
		hazard_rate = model->infected.infectious_curve[ model->time-1 - day ];
		n_infected =  model->infected.n_daily_current[ day];
		event = model->infected.events[ day ];
		for( idx = 0; idx < n_infected; idx++ )
		{
			infector      = event->individual;
			n_interaction = infector->n_interactions[ model->interaction_day_idx ];
			tot += n_interaction;

			interaction = infector->interactions[ model->interaction_day_idx ];
			for( jdx = 0; jdx < n_interaction; jdx++ )
			{
				if( interaction->individual->status == UNINFECTED )
				{
					interaction->individual->hazard -= hazard_rate;
					if( interaction->individual->hazard < 0 )
						new_infection( model, interaction->individual );
				}
				interaction = interaction->next;
			}
			event = event->next;
		}
	}
}

/*****************************************************************************************
*  Name:		transmit_virus
*  Description: Transmits virus over the interaction network
*  Returns:		void
******************************************************************************************/
void transmit_virus( model *model )
{
	transmit_virus_by_type( model, &(model->infected) );
	transmit_virus_by_type( model, &(model->symptomatic) );
}

/*****************************************************************************************
*  Name:		transition_infected
*  Description: Transitions infected who are due to become symptomatic
*  Returns:		void
******************************************************************************************/
void transition_infected( model *model )
{
	long idx, n_infected;
	int time_hospital;
	event *event;
	individual *indiv;

	n_infected = model->symptomatic.n_daily_current[ model->time ];
	event      = model->symptomatic.events[ model->time ];

	for( idx = 0; idx < n_infected; idx++ )
	{
		indiv = event->individual;
		remove_event_from_event_list( &(model->infected), indiv->current_event, indiv->time_infected );

		time_hospital = model->time + 2;
		indiv->current_event = add_individual_to_event_list( &(model->hospitalized), indiv, time_hospital, model );

		event = event->next;
	}
}

/*****************************************************************************************
*  Name:		add_indiv_to_event_list
*  Description: adds an individual to an event list at a particular time
*
*  Arguments:	list:	pointer to the event list
*  				indiv:	pointer to the individual
*  				time:	time of the event (int)
*  				model:	pointer to the model
*
*  Returns:		a pointer to the newly added event
******************************************************************************************/
event* add_individual_to_event_list(
	event_list *list,
	individual *indiv,
	int time,
	model *model
)
{
	event *event        = new_event( model );
	event->individual   = indiv;

	if( list->n_daily_current[time] > 1  )
	{
		list->events[ time ]->last = event;
		event->next  = list->events[ time ];
	}
	else
	{
		if( list->n_daily_current[time] == 1 )
		{
			list->events[ time ]->next = event;
			list->events[ time ]->last = event;
			event->next = list->events[ time ];
			event->last = list->events[ time ];
		}
	}

	list->events[time ] = event;
	list->n_daily[time]++;
	list->n_daily_current[time]++;

	return event;
}

/*****************************************************************************************
*  Name:		remove_event_from_event_list
*  Description: removes an event from an list at a particular time
*
*  Arguments:	list:	pointer to the event list
*  				event:	pointer to the event
*  				time:	time of the event (int)
*
*  Returns:		a pointer to the newly added event
******************************************************************************************/
void remove_event_from_event_list(
	event_list *list,
	event *event,
	int time
)
{
	if( list->n_daily_current[ time ] > 1 )
	{
		if( event != list->events[ time ] )
		{
			event->last->next = event->next;
			event->next->last = event->last;
		}
		else
			list->events[ time ] = event->next;
	}

	list->n_current--;
	list->n_daily_current[ time ]--;
}

/*****************************************************************************************
*  Name:		update_event_list_counters
*  Description: updates the event list counters, called at the end of a time step
*  Returns:		void
******************************************************************************************/
void update_event_list_counters( event_list *list, model *model )
{
	list->n_current += list->n_daily_current[ model->time ];
	list->n_total	+= list->n_daily[ model->time ];
}

/*****************************************************************************************
*  Name:		new_infection
*  Description: infects a new individual
*  Returns:		void
******************************************************************************************/
void new_infection( model *model, individual *indiv )
{
	int time_symptoms;
	indiv->status        = PRESYMPTOMATIC;
	indiv->time_infected = model->time;
	indiv->current_event = add_individual_to_event_list( &(model->infected), indiv, model->time, model );

	time_symptoms = model->time + sample_draw_list( model->symptomatic_draws );
	add_individual_to_event_list( &(model->symptomatic), indiv, time_symptoms, model );
}

/*****************************************************************************************
*  Name:		set_up_event_list
*  Description: sets up an event_list
*  Returns:		void
******************************************************************************************/
void set_up_event_list( event_list *list, parameters *params )
{
	int day;

	list->n_current = 0;
	list->n_total   = 0;
	for( day = 0; day < params->end_time;day ++ )
	{
		list->n_daily[day] = 0;
		list->n_daily_current[day] = 0;
	}
}

/*****************************************************************************************
*  Name:		set_up_seed_infection
*  Description: sets up the initial population
*  Returns:		void
******************************************************************************************/
void set_up_seed_infection( model *model )
{
	parameters *params = &(model->params);
	int idx;
	unsigned long int person;

	set_up_event_list( &(model->infected), params );

	for( idx = 0; idx < params->n_seed_infection; idx ++ )
	{
		person = gsl_rng_uniform_int( rng, params->n_total );
		new_infection( model, &(model->population[ person ]) );
	}
	update_event_list_counters( &(model->infected), model );
}

/*****************************************************************************************
*  Name:		build_daily_newtork
*  Description: Builds a new interaction network
******************************************************************************************/
void build_daily_newtork( model *model )
{
	long idx, n_pos;
	long *interactions = model->possible_interactions;
	long *all_idx = &(model->interaction_idx);

	interaction *inter1, *inter2;
	individual *indiv1, *indiv2;

	int day = model->interaction_day_idx;
	for( idx = 0; idx < model->params.n_total; idx++ )
		model->population[ idx ].n_interactions[ day ] = 0;

	n_pos = model->n_possible_interactions;
	gsl_ran_shuffle( rng, interactions, n_pos, sizeof(long) );

	idx = 0;
	n_pos--;
	while( idx < n_pos )
	{
		if( interactions[ idx ] == interactions[ idx + 1 ] )
		{
			idx++;
			continue;
		}

		inter1 = &(model->interactions[ (*all_idx)++ ]);
		inter2 = &(model->interactions[ (*all_idx)++ ]);
		indiv1 = &(model->population[ interactions[ idx++ ] ] );
		indiv2 = &(model->population[ interactions[ idx++ ] ] );

		inter1->individual = indiv2;
		inter1->next       = indiv1->interactions[ day ];
		indiv1->interactions[ day ] = inter1;
		indiv1->n_interactions[ day ]++;

		inter2->individual = indiv1;
		inter2->next       = indiv2->interactions[ day ];
		indiv2->interactions[ day ] = inter2;
		indiv2->n_interactions[ day ]++;

		if( *all_idx > model->n_interactions )
			*all_idx = 0;
	}
	fflush(stdout);
};

/*****************************************************************************************
*  Name:		one_time_step
*  Description: Move the model through one time step
******************************************************************************************/
int one_time_step( model *model )
{
	(model->time)++;
	build_daily_newtork( model );
	transmit_virus( model );
	transition_infected( model );

	update_event_list_counters( &(model->infected), model );
	update_event_list_counters( &(model->symptomatic), model );

	ring_inc( model->interaction_day_idx, model->params.days_of_interactions );
	return 1;
};

