/*
 * structure.h
 *
 *  Created on: 12 Mar 2020
 *      Author: p-robot
 * Description: structures used for the contact network
 */

#ifndef NETWORK_H_
#define NETWORK_H_

/************************************************************************/
/******************************* Includes *******************************/
/************************************************************************/

#include "structure.h"
#include "utilities.h"
#include "constant.h"
#include "params.h"

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

typedef struct network network;

struct edge{
	long id1;
	long id2;
};

struct network{
	edge *edges;	  			// array of edges
	long n_edges;	  			// number of edges in the network
	long n_vertices;  			// number of vertices
	int type;		  			// the type of network (.e. household/random/occupational)
	int skip_hospitalised;		// include the network for hospitalised people
	int skip_quarantined;		// include the network for quarantined people
	double daily_fraction;  	// fraction of the daily network sampled
	int network_id;				// unique network ID
	char name[INPUT_CHAR_LEN]; 	// unique name of the network

	int construction;			// method used to construct the network
	long opt_n_indiv;			// (OPTIONAL) number of distinct individuals on an network
	long *opt_pdx_array;		// (OPTIONAL) individual index of each person on the network
	int *opt_int_array;		    // (OPTIONAL) an integer associated with each individual
	long opt_long;				// (OPTIONAL) a long
	long *opt_long_array;	    // (OPTIONAL) an long array

	network *next_network;		// pointer to the next network
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

network* create_network(long n_total, int type);
void build_watts_strogatz_network( network *, long, double, double, int );
int check_member_or_self(long , long, long *, long );
void remove_contact(long *, long , long *);
void add_contact(long *, long , long *);
void relabel_network( network*, long*  );
void destroy_network( network* );
int update_daily_fraction( network*, double );

#endif /* NETWORK_H_ */
