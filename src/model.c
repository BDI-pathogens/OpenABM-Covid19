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

	return model;
};

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
*  Name:		one_time_step
*  Description: Move the model through one time step
******************************************************************************************/
int one_time_step( model *model )
{
	long idx, all_idx, n_pos;
	long interactions[ model->n_possible_interactions ];
	interaction *inter1, *inter2;
	individual *indiv1, *indiv2;

	int day = model->interaction_day_idx;
	for( idx = 0; idx < model->params.n_total; idx++ )
		model->population[ idx ].n_interactions[ day ] = 0;

	n_pos = model->n_possible_interactions;
	for( idx = 0; idx < n_pos; idx++ )
		interactions[ idx ] = model->possible_interactions[ idx ];
	gsl_ran_shuffle( rng, interactions, n_pos, sizeof(long) );

	idx = 0;
	all_idx = model->interaction_idx;
	n_pos--;
	while( idx < n_pos )
	{
		if( interactions[ idx ] == interactions[ idx + 1 ] )
		{
			idx++;
			continue;
		}

		inter1 = &(model->interactions[ all_idx++ ]);
		inter2 = &(model->interactions[ all_idx++ ]);
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
	}

	(model->time)++;
	ring_inc( model->interaction_day_idx, model->params.days_of_interactions );
	return 1;
};
