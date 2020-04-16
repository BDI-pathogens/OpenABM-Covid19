/*
 * hsopital.c
 *
 *  Created on: 16 Apr 2020
 *      Author: vuurenk
 */
#include <stdlib.h>
#include "waiting_list.h"

node* initialise_node( long pdx )
{
    node* node = malloc(sizeof(node));

    if (!node) 
        return NULL;

    node->pdx = pdx;
    node->next = NULL;
    return node;
}

// waiting_list* initialise_waiting_list()
// {
//     waiting_list* waiting_list = malloc( sizeof(waiting_list));
//     if( !waiting_list )
//         return NULL;
    
//     waiting_list->head = NULL;
//     waiting_list->size = 0;

//     return waiting_list;
// }   

void initialise_waiting_list( waiting_list *waiting_list )
{   
    waiting_list->head = NULL;
    waiting_list->size = 0;
}   

void push( long pdx, waiting_list *waiting_list )
{
    node *current = NULL;
    if( waiting_list->head == NULL )
    {
        waiting_list->head = initialise_node( pdx );
    } else {
        current = waiting_list->head; 
        
        while (current->next != NULL )
            current = current->next;
        
        current->next = initialise_node( pdx );
    }
    waiting_list->size++;
}

long pop( waiting_list* waiting_list )
{
    long retval;

    node* top = waiting_list->head;
    node* next = top->next;
    
    if( top == NULL )
        return -1;
    
    retval = top->pdx;
    top->next = NULL;
    
    waiting_list->head = next;
    waiting_list->size--;
    
    free( top );

    return retval;
}

void destroy_waiting_list( waiting_list* waiting_list )
{
    node* current = waiting_list->head;
    node* next = NULL;
    
    while( current != NULL )
    {
        next = current->next;
        free( current );
        current = next;
    }

    free( waiting_list );

}