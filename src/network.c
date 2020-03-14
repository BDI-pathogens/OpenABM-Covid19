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
#include <math.h>

#include "network.h"

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

	network_ptr->n_edges    = 0;
	network_ptr->n_vertices = n_total;
	
	return network_ptr;
}


/*****************************************************************************************
*  Name:		build_watts_strogatz_network
*  Description: Build a Watts Strogatz network
*
*  Arguments:	network  		- pointer to network which is constructed
*				N		 		- total number of nodes (long)
*				k		 		- mean number of connections (long)
*				p_rewire 		- probability of a connection being rewired (double)
*				randomise_nodes - put nodes on circular lattice randomly (int FALSE/TRUE)
*
*  Returns:		void
******************************************************************************************/
void build_watts_strogatz_network(
	network *network,
	long N,
	long k,
	double p_rewire,
	int randomise_nodes
)
{
	long incr = k/2, neighbour, i, j, l;
	
	// Allocate memory (needed for large N)
	long** edge_mat;
	edge_mat = calloc(N, sizeof(long *));
	for(i = 0; i < N; i++)
		edge_mat[i] = calloc(k*10, sizeof(long));
	
	// Degree for each individual
	int* n_edges_arr = calloc(N, sizeof(int));
	for(i = 0; i < N; i++){
		n_edges_arr[i] = k;
	}

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
				
				// Draw a new contact (long between 0 and N-1)
				new_contact = (long) floor(gsl_rng_uniform(rng) * N);
				
				// Check if new_connection is already connected (or is self)
				while(check_member_or_self(new_contact, i, edge_mat[i], n_edges_arr[i])){
					new_contact = gsl_rng_uniform_int(rng, N);
				}
				
				// Remove contact between person "i" and the original contact
				old_contact = edge_mat[i][j];
				
				remove_contact(edge_mat[old_contact], i, &(n_edges_arr[old_contact]));
				
				// if not, rewire to this new contact
				edge_mat[i][j] = new_contact;
				
				// Add i to list of connections to new_contact (and increase number of edges)
				add_contact(edge_mat[new_contact], i, &(n_edges_arr[new_contact]));
			}
		}
	}
	// Count total edges (i.e. network->n_edges)
	long n_edges = 0;
	for(i = 0; i < N; i++){
		n_edges += n_edges_arr[i];
	}
	network->n_edges = n_edges;
	
	// Form array of total edges (i.e. network->edges)
	network->edges = calloc(n_edges, sizeof(edge));

	// randomise the order nodes are put on the lattice if appropriate
	long* node_list = calloc(N, sizeof(long));
	for( i = 0; i < N; i++ )
		node_list[i] = i;
	if( randomise_nodes )
		gsl_ran_shuffle( rng, node_list, N, sizeof(long) );

	long idx = 0;
	for(i = 0; i < N; i++){
		for(j = 0; j < n_edges_arr[i]; j++){
			network->edges[idx].id1 = node_list[i];
			network->edges[idx].id2 = node_list[edge_mat[i][j]];
			idx++;
		}
	}
	
	for(i = 0; i < N; i++)
		free(edge_mat[i]);
	free(edge_mat);
	free(n_edges_arr);
	free(node_list);
};

/*****************************************************************************************
*  Name:		remove_contact
*  Description: Remove a contact from a list of edges, tidy list
******************************************************************************************/
void remove_contact(long *current_contacts_arr, long contact_to_remove, int *length){
	
	int i, j = 0;
	
	for(i = 0; i < *length; i++){
		if(current_contacts_arr[i] != contact_to_remove){
			current_contacts_arr[j] = current_contacts_arr[i];
			j++;
		}
	}
	current_contacts_arr[j] = UNKNOWN;
	--*length;
};

/*****************************************************************************************
*  Name:		add_contact
*  Description: Add a contact from a list of edges
******************************************************************************************/
void add_contact(long *current_contacts_arr, long contact_to_add, int *length){
	
	current_contacts_arr[*length] = contact_to_add;
	++*length;
};

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
};

/*****************************************************************************************
*  Name:		destroy_network
*  Description: Destroys the network structure and releases its memory
******************************************************************************************/
void destroy_network( network *network )
{
    free( network );
};

/*****************************************************************************************
*  Name:		relabel_network
*  Description: Takes a network with vertices 1..n and relabels them with longs t
*  				taken from list
******************************************************************************************/
void relabel_network( network *network, long *labels )
{
	long idx;
	for( idx = 0; idx < network->n_edges; idx++ )
	{
		network->edges[idx].id1 = labels[network->edges[idx].id1];
		network->edges[idx].id2 = labels[network->edges[idx].id2];
	}
};

