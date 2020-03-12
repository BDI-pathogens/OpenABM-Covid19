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
void destroy_network();

#endif /* NETWORK_H_ */
