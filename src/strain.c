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
	int idx,
	float transmission_multiplier
)
{
	strain *strain_ptr;
	strain_ptr = &(model->strains[ idx ]);
	if( strain_ptr->idx != 0 )
		print_exit( "Strains can only be intitialised once!" );
	strain_ptr->idx 					= idx;
	strain_ptr->transmission_multiplier = transmission_multiplier;
	model->n_initialised_strains++;
	model->cross_immunity[idx][idx] = 1;
}