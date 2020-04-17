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

void initialise_waiting_list( waiting_list *waiting_list )
{   
    waiting_list->head = NULL;
    waiting_list->size = 0;
}   

long pdx_at( waiting_list* waiting_list, int idx )
{
    int i = 0;
    node* current = waiting_list->head;
    
    while( i < idx )
        current = current->next;

    return current->pdx;
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
        return WAITING_LIST_EMPTY;
    
    retval = top->pdx;
    free( top );

    waiting_list->head = next;
    waiting_list->size--;

    return retval;
}

void remove_patient(long pdx, waiting_list* waiting_list)
{
    node *current = waiting_list->head;            
    node *previous = current;

    while( current != NULL )
    {           
        if( current->pdx == pdx )
        {      
            previous->next = current->next;

            if( current == waiting_list->head )
            waiting_list->head = current->next;
            
            free( current );
            return;
        }                               
        previous = current;             
        current = current->next;        
    }                                 
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