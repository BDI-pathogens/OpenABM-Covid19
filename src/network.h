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

struct edge{
	long id1;
	long id2;
};

typedef struct{
	edge *edges;	// array of edges
	long n_edges;	// number of edges in the network
} network;

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

network* new_network(long n_total);
void build_watts_strogatz_network( network *, parameters * );
int check_member_or_self(long , long, long *, int );
void remove_contact(long *, long , int *);
void add_contact(long *, long , int *);
void destroy_network();

#endif /* NETWORK_H_ */
