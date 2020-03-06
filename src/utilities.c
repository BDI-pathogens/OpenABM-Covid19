/*
 * utilities.c
 *
 *  Created on: 5 Mar 2020
 *      Author: hinchr
 */

#include <stdio.h>
#include <stdlib.h>

/*****************************************************************************************
*  Name:		print_exit
******************************************************************************************/
void print_exit( char *s )
{
    printf("%s\n", s );
    fflush(stdout);
    exit(1);
}

