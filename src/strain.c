/*
 * strain.c
 *
 *  Created on: 11 Mar 2021
 *      Author: nikbaya
 */

#include "strain.h"
#include "model.h" // for rng
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
	float transmission_multiplier
)
{
	new->idx 						= idx;
	new->parent 					= parent;
	new->transmission_multiplier 	= transmission_multiplier;
}


// /*****************************************************************************************
// *  Name:		--
// *  Description: --
// *  Returns:		void
// ******************************************************************************************/
// void set_strain_multiplier(
// 	strain *strain,
// 	float transmission_multiplier
// )
// {
// 	strain->transmission_multiplier = transmission_multiplier;
// }


// /*****************************************************************************************
// *  Name:		--
// *  Description: --
// *  Returns:		void
// ******************************************************************************************/
// void set_parent_idx(
// 	strain *strain,
// 	long parent_idx
// )
// {
// 	strain->parent_idx = parent_idx;
// }


/*****************************************************************************************
*  Name:		--
*  Description: --
*  Returns:		--
******************************************************************************************/
// void mutate_strain(
// 	strain *parent
// )
// {	
// 	double mutation_prob = 0.50;
// 	double mutation_draw = gsl_rng_uniform( rng );

// 	// if( mutation_draw < mutation_prob )
// 	// {
// 	// 	long child_idx = parent-idx + 1
// 	// 	long parent_idx = parent->idx;
// 	// 	float child_transmission_multiplier = parent->transmission_multiplier*1.5;
// 	// 	printf("Mutation: parent: %ld, %f\n", parent_idx, child_transmission_multiplier);
// 	// 	initialize_strain( child, child_idx, parent_idx, child_transmission_multiplier );
// 	// }


// }