/*
 * model.c
 *
 *  Created on: 5 Mar 2020
 *      Author: hinchr
 */

#include "model.h"
#include "individual.h"
#include "constant.h"
#include "params.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

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
	model *model  = malloc( sizeof( model ) );
	model->params = *params;

	set_up_gsl( model );
	set_up_population( model );
	set_up_interactions( model );

	return model;
};

/*****************************************************************************************
*  Name:		set_up_gsl
*  Description: sets up gsl library
*  Returns:		void
******************************************************************************************/
void set_up_gsl( model *model )
{
	gsl_rng_env_setup();
	model->rng = malloc( sizeof( gsl_rng ) );
	gsl_rng *rng = gsl_rng_alloc( gsl_rng_default );
	model->rng = rng;
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

	model->population = malloc( params->n_total * sizeof( individual ) );
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
	long idx, n_idx, indiv_idx;

	// FIXME - need to a good estimate of the total number of interactions
	//         easy at the moment since we have a fixed number per individual
	long n_daily_interactions = params->n_total * params->mean_daily_interactions;
	long n_interactions       = n_daily_interactions * params->days_of_interactions;

	model->interactions          = malloc( n_interactions * sizeof( interaction ) );
	model->n_interactions        = n_interactions;
	model->interaction_idx       = 0;
	model->interaction_day_idx   = 0;

	model->possible_interactions = malloc( n_daily_interactions * sizeof( long ) );
	idx = 0;
	for( indiv_idx = 0; indiv_idx < params->n_total; indiv_idx++ )
		for( n_idx = 0; n_idx < model->population[ indiv_idx ].n_interactions; n_idx++ )
			model->possible_interactions[ idx++ ] = indiv_idx;
	model->n_possible_interactions = idx;
}

/*****************************************************************************************
*  Name:		destroy_model
*  Description: Destroys the model structure and releases its memory
*
******************************************************************************************/
void destroy_model( model *model )
{
    gsl_rng_free( model->rng );

    free( model->population );
 //   free( model->possible_interactions );
    free( model->interactions );

};
