/*
 * network.c
 *
 *  Created on: 12 Mar 2020
 *      Author: p-robot
 */

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

#include "network.h"
#include "utilities.h"
#include "constant.h"
#include "params.h"

/*****************************************************************************************
*  Name:		new_network
*  Description: Builds a new model object from a number of individuals
*  				 1. Creates memory for it
*  Returns:		pointer to model
******************************************************************************************/


network* new_network(long n_total)
{	
	network *network_ptr = NULL;
	network_ptr = calloc( 1, sizeof( network ) );
	if( network_ptr == NULL )
    	print_exit("calloc to network failed\n");

	network_ptr->n_edges = 0;
	
	return network_ptr;
}



/*****************************************************************************************
*  Name:		destroy_network
*  Description: Destroys the network structure and releases its memory
******************************************************************************************/
void destroy_network( network *network )
{
    free( network );
};

