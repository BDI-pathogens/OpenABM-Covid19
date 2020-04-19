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

//TODO: rename to list?
struct waiting_list 
{
    node* head;
    int size;
};

node* initialise_node( long pdx );
void initialise_waiting_list( waiting_list *waiting_list );

//TODO: rename these. maybe put list_ in front of each name to make clear?
long pdx_at( waiting_list* waiting_list, int idx );
void push_front( long pdx, waiting_list *waiting_list );
void push_back( long pdx, waiting_list *waiting_list );
long pop( waiting_list* waiting_list );
void remove_patient( long pdx, waiting_list* waiting_list);
int  list_elem_exists( long pdx, waiting_list *list );
void destroy_waiting_list( waiting_list* waiting_list );

#endif