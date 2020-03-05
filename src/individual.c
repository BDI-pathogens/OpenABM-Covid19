/*
 * individual.c
 *
 *  Created on: 5 Mar 2020
 *      Author: hinchr
 */

#include "individual.h"
#include "params.h"
#include "constant.h"

/*****************************************************************************************
*  Name:		initialize_individual
*  Description: initializes and individual at the start of the simulation
*  Returns:		void
******************************************************************************************/
void initialize_individual(
	individual *indiv,
	parameters *params
)
{
	indiv->status = UNINFECTED;
	indiv->n_interactions = params->mean_daily_interactions;
}

