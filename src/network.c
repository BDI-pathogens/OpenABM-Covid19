/*
 * network.c
 *
 *  Created on: 12 Mar 2020
 *      Author: p-robot
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "network.h"

/*****************************************************************************************
*  Name:		create_network
*  Description: Builds a new model object from a number of individuals
*  				 1. Creates memory for it
*  Returns:		pointer to model
******************************************************************************************/
network* create_network( long n_total, int type )
{	
	network *network_ptr = NULL;
	network_ptr = calloc( 1, sizeof( network ) );
	if( network_ptr == NULL )
    	print_exit("calloc to network failed\n");

	network_ptr->n_edges    = 0;
	network_ptr->n_vertices = n_total;
	network_ptr->type       = type;
	
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
	double k,
	double p_rewire,
	int randomise_nodes
)
{
    // Check upper-bound of k and potentially reset it.
    // Upper-bound of k is N-1 if N is odd, N-2 if N is even.
    if ( 2 * ceil( k / 2 )  >= N - 1 ) k = ( N - 1) / 2 * 2;

    // If k is the upper-bound value, no need to rewire.
    if ( k ==  (N - 1) / 2 * 2 ) p_rewire = 0.0;

	long k_used, k_right, i, j, ii;
	double p_right;

	// handle non-integer k, have different number of connections to the right
	k_right = floor( k / 2 );
	p_right = k / 2 -  k_right;

	// Allocate memory (needed for large N)
	long** edge_mat;
	edge_mat = calloc( N, sizeof(long *) );
	for(i = 0; i < N; i++)
		edge_mat[i] = calloc( ceil( k*10 ), sizeof(long) );
	
	// Degree for each individual (need to store a copy during the first step
	long* n_edges_arr      = calloc( N, sizeof(long) );
	long* n_edges_arr_init = calloc( N, sizeof(long) );

	// Step 1: Set up random lattice
	// start by getting the correct number interactions but only mark only the connections to the right
	for(i = 0; i < N; i++)
	{
		k_used = k_right + gsl_ran_bernoulli( rng, p_right );
		n_edges_arr[i]      = k_used;
		n_edges_arr_init[i] = k_used;

		for( j = 0; j < k_used; j++ )
			edge_mat[i][j] = ( i + j + 1 + N) % N;
	}
	// now mark all the connections to the left and get the total interactions per node
	for(i = 0; i < N; i++)
		for( j = 0; j < n_edges_arr_init[i]; j++ )
		{
			ii = edge_mat[i][j];
			edge_mat[ii][n_edges_arr[ii]++] = i;
		}
	free( n_edges_arr_init );

	double u;
	long new_contact, old_contact;

	// Step 2: Randomly rewire connections with probability "p_rewire"
	for(i = 0; i < N; i++){
		for(j = 0; j < n_edges_arr[i]; j++){

			u = gsl_rng_uniform(rng);
			
			if(u < p_rewire){
				
				// Draw a new contact (long between 0 and N-1)
				new_contact =  gsl_rng_uniform_int(rng, N);
				
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
	// Count total edges (i.e. network->n_edges) in undirected graph
	// (above code constructs a directed graph)
	long n_edges = 0;
	for(i = 0; i < N; i++){
		n_edges += n_edges_arr[i];
	}
	network->n_edges = n_edges/2;
	
	// Form array of total edges (i.e. network->edges)
	network->edges = calloc(n_edges, sizeof(edge));

	// randomise the order nodes are put on the lattice if appropriate
	long* node_list = calloc(N, sizeof(long));
	for( i = 0; i < N; i++ )
		node_list[i] = i;
	if( randomise_nodes )
		gsl_ran_shuffle( rng, node_list, N, sizeof(long) );

	long idx = 0;
	for(i = 0; i < N; i++)
		for(j = 0; j < n_edges_arr[i]; j++)
		{
			network->edges[idx].id1 = node_list[i];
			network->edges[idx].id2 = node_list[edge_mat[i][j]];
			idx++;
			
			// Remove connections between id2 -> id1 
			// (otherwise we count edges twice)
			remove_contact(edge_mat[edge_mat[i][j]],
				i, &(n_edges_arr[edge_mat[i][j]]));
		}
	
	for(i = 0; i < N; i++)
		free(edge_mat[i]);
	free(edge_mat);
	free(n_edges_arr);
	free(node_list);
}

/*****************************************************************************************
*  Name:		remove_contact
*  Description: Remove a contact from a list of edges, tidy list
******************************************************************************************/
void remove_contact(long *current_contacts_arr, long contact_to_remove, long *length)
{
	long i, j = 0;
	
	for(i = 0; i < *length; i++){
		if(current_contacts_arr[i] != contact_to_remove){
			current_contacts_arr[j] = current_contacts_arr[i];
			j++;
		}
	}
	current_contacts_arr[j] = UNKNOWN;
	--*length;
}

/*****************************************************************************************
*  Name:		add_contact
*  Description: Add a contact from a list of edges
******************************************************************************************/
void add_contact(long *current_contacts_arr, long contact_to_add, long *length)
{
	current_contacts_arr[*length] = contact_to_add;
	++*length;
}

/*****************************************************************************************
*  Name:		check_member_or_self
*  Description: Check if x is 'self' or a member of the 'array' (of length 'length')
******************************************************************************************/
int check_member_or_self(long x, long self, long *array, long length)
{
	if(x == self){
		return 1;
	}
	
    long i;
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
	free( network->edges );

	if( network->opt_pdx_array != NULL )
		free( network->opt_pdx_array );

	if( network->opt_int_array != NULL )
		free( network->opt_int_array );

	if( network->opt_long_array != NULL )
		free( network->opt_long_array );

	free( network );
}

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
}

/*****************************************************************************************
*  Name:		update_daily_fraction
*  Description: Updates the daily_fraction on the network
******************************************************************************************/
int update_daily_fraction( network *network, double fraction )
{
	if( fraction < 0 || fraction > 1 )
		return FALSE;

	if( ( network->construction == NETWORK_CONSTRUCTION_RANDOM_DEFAULT ) ||
		( network->construction == NETWORK_CONSTRUCTION_RANDOM ) )
	{
		if( ( fraction > 1e-9 ) && ( fraction <  1 - 1e-9 ) )
			return FALSE;
	}

	network->daily_fraction = fraction;
	return TRUE;
}


