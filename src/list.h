/*
 * list.h
 *
 *  Created on: 16 Apr 2020
 *      Author: vuurenk
 */

#ifndef LIST_H_
#define LIST_H_

#define WAITING_LIST_EMPTY -1

typedef struct node node;

struct node 
{
    long data;
    struct node *next;
};

node* initialise_node( long data );

typedef struct list list;

struct list
{
    node* head;
    int size;
};

void initialise_list( list *list );
long list_element_at( list* list, int index );
int  list_elem_exists( long pdx, list *list );
void list_push_front( long data, list *list );
void list_push_back( long data,  list *list );
void list_remove_element( long data, list *list);
long list_pop( list* list );
void destroy_list( list* list );

#endif
