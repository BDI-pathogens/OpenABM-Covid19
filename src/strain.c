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
*  Name:		add_new_strain
*  Description: Initialises new strain, can only be called once for each strain
*  Returns:		void
******************************************************************************************/
short add_new_strain(
	model *model,
	float transmission_multiplier
)
{
	strain *strain_ptr;

	strain_ptr = &(model->strains[ model->n_initialised_strains ]);
	strain_ptr->idx 					= model->n_initialised_strains;
	strain_ptr->transmission_multiplier = transmission_multiplier;

	model->n_initialised_strains++;
	return(  model->n_initialised_strains - 1 );
}
