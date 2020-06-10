/*
 * list.c
 *
 *  Created on: 16 Apr 2020
 *      Author: vuurenk
 */
#include <stdlib.h>
#include "list.h"
#include "constant.h"

node* initialise_node( long data )
{
	node* node = malloc(sizeof(node));

	if (!node) 
		return NULL;

	node->data = data;
	node->next = NULL;
	return node;
}

void initialise_list( list *list )
{   
	list->head = NULL;
	list->size = 0;
}   

long list_element_at( list* list, int index )
{
	int i = 0;
	node* current = list->head;
	
	while( i < index )
	{
		current = current->next;
		i++;
	}

	return current->data;
}

void list_push_front( long data, list *list )
{
	node* current = NULL;

	if( list->head == NULL )
	{
		list->head = initialise_node( data );
	}
	else 
	{
		current = list->head;
		list->head = initialise_node( data );
		list->head->next = current;
	}
	
	list->size++;
}

void list_push_back( long data, list *list )
{
	node *current = NULL;

	if( list->head == NULL )
		list->head = initialise_node( data );
	else 
	{
		current = list->head;
		
		while (current->next != NULL )
			current = current->next;
		
		current->next = initialise_node( data );
	}
	list->size++;
}

int list_elem_exists( long data, list *list )
{
	if( list->head == NULL )
		return FALSE;
	
	node* current = list->head;

	while( current != NULL )
	{           
		if( current->data == data )
			return TRUE;

		current = current->next;        
	} 

	return FALSE;
}

long list_pop( list* list )
{
	long retval;

	node* top = list->head;
	node* next = NULL;
	
	if( top == NULL )
		return WAITING_LIST_EMPTY;
	
	next = top->next;
	
	retval = top->data;
	free( top );

	list->head = next;
	list->size--;

	return retval;
}

void list_remove_element( long data, list* list )
{
	if( list->head == NULL )
		return;

	node *current = list->head;
	node *previous = current;

	while( current != NULL )
	{
		if( current->data == data )
		{
			previous->next = current->next;

			if( current == list->head )
				list->head = current->next;

			free( current );
			list->size--;
			return;
		}
		previous = current;
		current = current->next;
	}
}

void destroy_list( list* list )
{
	node* current = list->head;
	node* next = NULL;
	
	while( current != NULL )
	{
		next = current->next;
		free( current );
		current = next;
	}
}
