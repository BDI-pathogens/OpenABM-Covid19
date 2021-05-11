/*
 * strain.c
 *
 *  Created on: 31 Mar 2021
 *      Author: nikbaya
 */

#include "strain.h"
#include "model.h"
#include "utilities.h"

/*****************************************************************************************
*  Name:		initialise_strain
*  Description: --
*  Returns:		--
******************************************************************************************/
void initialise_strain(
	model *model,
	long idx,
	float transmission_multiplier,
	float time_multiplier
)
{
	strain *strain_ptr;
	strain_ptr = &(model->strains[ idx ]);
	if( strain_ptr->idx != -1 )
		print_exit( "Strains can only be initialised once!" );
	strain_ptr->idx 					= idx;
	strain_ptr->parent_idx				= -1;
	strain_ptr->transmission_multiplier = transmission_multiplier;
	strain_ptr->antigen_phen 			= calloc( ANTIGEN_PHEN_DIM, sizeof( double ) );
	model->n_initialised_strains++;

	// if( idx < MAX_N_STRAINS-1 )
	// {
	// 	strain *mutant;
	// 	mutant = mutate_strain( model, strain_ptr );
	// }
	// else
	// 	printf("idx%ld\n", idx);
}

/*****************************************************************************************
*  Name:		set_antigen_phen_distance
*  Description: --
*  Returns:		void
******************************************************************************************/
void set_antigen_phen_distance( 
	model *model, 
	long strain_idx1, 
	long strain_idx2,
	float distance 
)
{
	int idx = get_triu_idx( strain_idx1, strain_idx2 );
	model->antigen_phen_distances[ idx ] = distance;
}

/*****************************************************************************************
*  Name:		get_antigen_phen_distance
*  Description: --
*  Returns:		float
******************************************************************************************/
float get_antigen_phen_distance( 
	model *model,
	strain *strain1,
	strain *strain2
)
{
	if( strain1 == strain2 )
		return 0.0;
	
	int idx = get_triu_idx( strain1->idx, strain2->idx );
	if( model->antigen_phen_distances[ idx ] == -1.0 ) // distance has not been calculated
	{
		float distance = 0;
		for( int dim = 0; dim < ANTIGEN_PHEN_DIM; dim++ )
			distance += pow( strain1->antigen_phen[dim] - strain2->antigen_phen[dim], 2 );
		distance = sqrt( distance );
		model->antigen_phen_distances[ idx ] = distance;
	}

	return model->antigen_phen_distances[ idx ];
}