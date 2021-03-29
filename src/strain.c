/*
 * strain.c
 *
 *  Created on: 11 Mar 2021
 *      Author: nikbaya
 */

#include "strain.h"
#include "individual.h"
#include "model.h" // for rng
#include "constant.h" // for N_STRAIN_BINS constant
#include <stdio.h> // for printf


/*****************************************************************************************
*  Name:		--
*  Description: --
*  Returns:		void
******************************************************************************************/
void initialize_strain(
	strain *new,
	long idx,
	strain *parent,
	float transmission_multiplier,
	long n_infected,
	long total_infected

)
{
	new->idx 						= idx;
	new->parent 					= parent;
	new->transmission_multiplier 	= transmission_multiplier;
	new->n_infected					= n_infected;
	new->total_infected				= total_infected;
	new->next						= NULL;
}

/*****************************************************************************************
*  Name:		--
*  Description: --
*  Returns:		--
******************************************************************************************/
void mutate_strain(
	model *model,
	individual *infector,
	double mutation_prob
)
{	
	if( mutation_prob == 0 )
		return;
	
	double mutation_draw = gsl_rng_uniform( rng ); // Note that when we add a new call to the rng, we can't count on the downstream results being exactly as the original version

	if( mutation_draw < mutation_prob )
	{
		strain *parent 		= infector->infection_events->strain;
		strain *mutated 	= calloc( 1, sizeof( struct strain ) );
		long mutated_idx	= parent->idx + 1;
		
		double sigma 					= 0.5;  // stdev for distribtion of mutated strain's transmission_multiplier
		float delta 					= gsl_ran_gaussian( rng, sigma ); // change in transmission_multiplier due to mutation
		float transmission_multiplier 	= 1; //max(0, parent->transmission_multiplier + delta); // new transmission_multiplier for mutation

		// printf("Mutation!: %p -> %p\n", parent, mutated);
		// printf("\t%f -> %f\n", parent->transmission_multiplier, transmission_multiplier);

		initialize_strain( mutated, mutated_idx, parent, transmission_multiplier, 1, 1);
		add_new_strain_to_model( model, mutated );
		add_to_strain_bin_count( model->strain_bins, mutated->transmission_multiplier, 1);
		// add_to_strain_bin_count( model->strain_bins, parent->transmission_multiplier, -1);
		infector->infection_events->strain = mutated;
		parent->n_infected--;
	}
}

/*****************************************************************************************
*  Name:		--
*  Description: --
*  Returns:		--
******************************************************************************************/
void add_new_strain_to_model(
	model *model,
	strain *new
)
{
	strain *temp = model->strains;
	if( temp == NULL )
	{
		model->strains = new;
	}
	else
	{
		while( temp->next != NULL )
		{
			temp = temp->next;
		}
		temp->next = new;
	}
	model->n_strains++;
	model->total_transmission_multiplier += new->transmission_multiplier;
    
}

/*****************************************************************************************
*  Name:		--
*  Description: --
*  Returns:		--
******************************************************************************************/
void add_to_strain_bin_count(
	long* strain_bins,
	float new_value,
	int delta
)
{
	int bin_idx;
    float max_bin_lower_lim = 2; // lower limit of the maximum bin
    if( new_value < max_bin_lower_lim )
       	bin_idx = floor(new_value * (N_STRAIN_BINS - 1) / max_bin_lower_lim);
    else
    	bin_idx = N_STRAIN_BINS - 1;
    strain_bins[bin_idx] += delta;
}