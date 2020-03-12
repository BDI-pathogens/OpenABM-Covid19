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

void build_watts_strogatz_network( )
{
	long k = 10;
	long N = 1000;
	double p_rewire = 0.1;
	
	long incr = k/2, neighbour, i, j, l;
	
	printf("k : %li\n", k);
	printf("N : %li\n", N);
	printf("p_rewire : %f\n", p_rewire);
	
	// Allocate memory (needed for large N)
    long** edge_mat;
	edge_mat = calloc(N, sizeof(long *));
    for(i = 0; i < N; i++)
        edge_mat[i] = calloc(k, sizeof(long));
	
	// Degree for each individual
	int* n_edges_arr = calloc(N,  sizeof(int));
	for(i = 0; i < N; i++)
		n_edges_arr[i] = k;
	
	// Step 1: Set up random lattice
	for(i = 0; i < N; i++){
		j = 0; l = 0;
		while(l < k){
			// Make sure we loop to the start of the ring
			neighbour = (i - incr + j + N) % N;

			if(neighbour != i){
				edge_mat[i][l] = neighbour;
				l++;
			}
			j++;
		}
	}
	
	double u;
	long new_contact, old_contact;
	// Step 2: Randomly rewire connections with probability "p_rewire"
	for(i = 0; i < N; i++){
		for(j = 0; j < n_edges_arr[i]; j++){
			
			u = gsl_rng_uniform(rng);
			
			if(u < p_rewire){
				
				new_contact = i;
				
				// Check if new_connection is already connected (or is self)
				while(check_member_or_self(new_contact, i, edge_mat[i], n_edges_arr[i])){
					new_contact = gsl_rng_uniform_int(rng, N);
				}
				
				old_contact = edge_mat[i][j];
				
				// if not, rewire to this new contact
				edge_mat[i][j] = new_contact;
				
				// rewire other person too
				remove_contact(edge_mat[old_contact], i, &(n_edges_arr[old_contact]));
			}
		}
	}
	
	// Count total edges (i.e. network->n_edges)
	long n_edges = 0;
	for(i = 0; i < N; i++){
		n_edges += n_edges_arr[i];
	}
	
	// Form array of total edges (i.e. network->edges)
	edge* edges;
	edges = calloc(n_edges, sizeof(edge));
	long idx = 0;
	for(i = 0; i < N; i++){
		for(j = 0; j < n_edges_arr[i]; j++){
			edges[idx].id1 = i;
			edges[idx].id2 = edge_mat[i][j];
			idx++;
		}
	}
	
    for(i = 0; i < N; i++)
        free(edge_mat[i]);
	free(edge_mat);
	free(n_edges_arr);
	
	free(edges);
	
}


/*****************************************************************************************
*  Name:		remove_contact
*  Description: Remove a contact from a list of edges, tidy list
******************************************************************************************/

void remove_contact(long *current_contacts, long contact_to_remove, int *n_edges){
	int i, j = 0;
	for(i = 0; i < *n_edges; i++){
		
		if(current_contacts[i] != contact_to_remove){
			current_contacts[j] = current_contacts[i];
			j++;
		}
	}
	current_contacts[j] = UNKNOWN;
	--*n_edges;
}


/*****************************************************************************************
*  Name:		check_member_or_self
*  Description: Check if x is 'self' or a member of the 'array' (of length 'length')
******************************************************************************************/

int check_member_or_self(long x, long self, long *array, int length)
{
	if(x == self){
		return 1;
	}
	
    int i;
    for(i = 0; i < length; i++)
    {
        if(array[i] == x)
            return 1;
    }
    return 0;
}



/*****************************************************************************************
*  Name:		destroy_network
*  Description: Destroys the network structure and releases its memory
******************************************************************************************/
void destroy_network( network *network )
{
    free( network );
};

