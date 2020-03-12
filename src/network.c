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
*  Name:		watts_strogatz_network
*  Description: Build a Watts Strogatz network
*  Returns:		pointer to model
******************************************************************************************/

void build_watts_strogatz_network()
{
	
	// Output: network->n_edges
	// Output: network->edges
	
	long k = 10;
	long N = 1000;
	double p_rewire = 0.1;
	
	long incr = k/2, neighbour, i, j, l;
	
	printf("k : %li\n", k);
	printf("N : %li\n", N);
	printf("p_rewire : %f\n", p_rewire);
	
	// Allocate memory (needed for large N)
    long** edges = calloc(N,  sizeof(long*));
    for(i = 0; i < N; i++)
        edges[i] = calloc(k, sizeof(long));
	
	// Step 1: Set up random lattice
	
	for(i = 0; i < N; i++){
		j = 0; l = 0;
		while(l < k)
		{
			// Make sure we loop to the start of the ring
			neighbour = (i - incr + j + N) % N;
			
			if(neighbour != i){
				edges[i][l] = neighbour;
				l++;
			}
			j++;
		}
	}
	
	
	for(i = 0; i < N; i++){
		printf("%li ", i);
		for(j = 0; j < k; j++){
				printf("%li ", edges[i][j]);
		}
		printf("\n");
	}
	
	// Step 2: Randomly rewire connections
}

/*****************************************************************************************
*  Name:		destroy_network
*  Description: Destroys the network structure and releases its memory
******************************************************************************************/
void destroy_network( network *network )
{
    free( network );
};

