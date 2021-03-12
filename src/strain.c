/*
 * strain.c
 *
 *  Created on: 11 Mar 2021
 *      Author: nikbaya
 */

#include "strain.h"

/*****************************************************************************************
*  Name:		--
*  Description: --
*  Returns:		void
******************************************************************************************/
void initialize_strain(
	strain *strain,
	int idx,
	int parent_idx,
	float strain_multiplier
)
{
	strain->idx 				= idx;
	strain->parent_idx 			= parent_idx;
	strain->strain_multiplier 	= strain_multiplier;

}


/*****************************************************************************************
*  Name:		--
*  Description: --
*  Returns:		void
******************************************************************************************/
void set_strain_multiplier(
	strain *strain,
	float strain_multiplier
)
{
	strain->strain_multiplier = strain_multiplier;
}


/*****************************************************************************************
*  Name:		--
*  Description: --
*  Returns:		void
******************************************************************************************/
void set_parent_idx(
	strain *strain,
	long parent_idx
)
{
	strain->parent_idx = parent_idx;
}
