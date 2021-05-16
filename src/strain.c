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
*  Description: Initialises new strain, can only be called once for each strain
*  Returns:		void
******************************************************************************************/
void initialise_strain(
	model *model,
	long idx,
	float transmission_multiplier
)
{
	strain *strain_ptr;
	strain_ptr = &(model->strains[ idx ]);
	if( strain_ptr->idx != -1 )
		print_exit( "Strains can only be initialised once!" );
	strain_ptr->idx 					= idx;
	strain_ptr->transmission_multiplier = transmission_multiplier;
	model->n_initialised_strains++;
}