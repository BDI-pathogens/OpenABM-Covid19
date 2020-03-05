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

	gsl_rng_env_setup();
	model->rng = malloc( sizeof( gsl_rng ) );
	gsl_rng *rng = gsl_rng_alloc( gsl_rng_default );
	model->rng = rng;

	set_up_population( model );

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

	model->population = malloc( params->n_total * sizeof( individual ) );
	for( idx = 0; idx < params->n_total; idx++ )
		initialize_individual( &(model->population[idx]), params );

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
    free( model );
};
