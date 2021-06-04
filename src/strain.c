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
	float transmission_multiplier,
	double *hospitalised_fraction
)
{
	strain *strain_ptr;

	if( model->n_initialised_strains == model->params->max_n_strains )
		return ERROR;

	strain_ptr = &(model->strains[ model->n_initialised_strains ]);
	strain_ptr->idx 					= model->n_initialised_strains;
	strain_ptr->transmission_multiplier = transmission_multiplier;

	for( int idx = 0; idx < N_AGE_GROUPS; idx++ )
		strain_ptr->hospitalised_fraction[ idx ] = hospitalised_fraction[ idx ];

	model->n_initialised_strains++;
	return(  model->n_initialised_strains - 1 );
}

/*****************************************************************************************
*  Name:		get_strain_by_id
*  Description: returns a pointer to a strain a given ID
*  Returns:		pointer to vaccine
******************************************************************************************/
strain* get_strain_by_id( model *model, short strain_idx )
{
	if( strain_idx >=  model->n_initialised_strains )
		print_exit( "strain not yet intialised " );

	return &(model->strains[ strain_idx ]);
}
