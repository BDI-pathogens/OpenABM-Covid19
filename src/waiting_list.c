/*
 * waiting_list.c
 *
 *  Created on: 16 Apr 2020
 *      Author: vuurenk
 */
#include <stdlib.h>
#include "waiting_list.h"
#include "constant.h"

/*****************************************************************************************
*
* Generic functions for creating a linked list. Used primarily for hospital wards.
*
******************************************************************************************/
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
    //waiting_list = calloc( 1, sizeof(waiting_list));
    waiting_list->head = NULL;
    waiting_list->size = 0;
}   

long pdx_at( waiting_list* waiting_list, int idx )
{
    int i = 0;
    node* current = waiting_list->head;
    
    while( i < idx )
    {
        current = current->next;
        i++;
    }

    return current->pdx;
}

void push_front( long pdx, waiting_list *waiting_list )
{
    node* current = NULL;

    if( waiting_list->head == NULL )
    {
        waiting_list->head = initialise_node( pdx );
    }
    else 
    {
        current = waiting_list->head;
        waiting_list->head = initialise_node( pdx );
        waiting_list->head->next = current;
    }
    
    waiting_list->size++;
}

void push_back( long pdx, waiting_list *waiting_list )
{
    node *current = NULL;

    if( waiting_list->head == NULL )
        waiting_list->head = initialise_node( pdx );
    else 
    {
        current = waiting_list->head; 
        
        while (current->next != NULL )
            current = current->next;
        
        current->next = initialise_node( pdx );
    }
    waiting_list->size++;
}

int list_elem_exists( long pdx, waiting_list *list )
{
    if( list->size == 0 || list->head == NULL )
        return FALSE;
    
    node* current = list->head;

    while( current != NULL )
    {           
        if( current->pdx == pdx )
        {      
            return TRUE;
        }             
        current = current->next;        
    } 

    return FALSE;
}

long pop( waiting_list* waiting_list )
{
    long retval;

    node* top = waiting_list->head;
    node* next = NULL;
    
    if( top == NULL )
        return WAITING_LIST_EMPTY;
    
    if( waiting_list->size > 1 )
        next = top->next;
    
    retval = top->pdx;
    free( top );

    waiting_list->head = next;
    waiting_list->size--;

    return retval;
}

void remove_patient(long pdx, waiting_list* waiting_list)
{
    if( waiting_list->head == NULL )
        return;

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
            waiting_list->size--;                                
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

    //free( waiting_list );
}
