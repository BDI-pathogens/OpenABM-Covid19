/*
 * waiting_list.h
 *
 *  Created on: 16 Apr 2020
 *      Author: vuurenk
 */

#ifndef WAITING_LIST_H_
#define WAITING_LIST_H_

#define WAITING_LIST_EMPTY -1

typedef struct node node;

struct node 
{
    long pdx;
    struct node *next;
};

typedef struct waiting_list waiting_list;

struct waiting_list 
{
    node* head;
    int size;
};

node* initialise_node( long pdx );
void initialise_waiting_list( waiting_list *waiting_list );

void push( long pdx, waiting_list *waiting_list );
long pop( waiting_list* waiting_list );
void remove( long pdx, waiting_list* waiting_list);

void destroy_waiting_list( waiting_list* waiting_list );

#endif