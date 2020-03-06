// main.c

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

#include "model.h"
#include "params.h"
#include "utilities.h"
#include "input.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

int main(int argc, char *argv[])
{
    printf("Starting simulation\n");

    parameters params;	
	
	printf("Read input parameter file\n");
	read_param_file( &params );
	
    if( params.days_of_interactions > MAX_DAILY_INTERACTIONS )
    	print_exit( "asking for day_of_interaction to be greater than MAX_DAILY_INTERACTIONS " );
	
    gsl_rng_env_setup();
    rng = gsl_rng_alloc ( gsl_rng_default);
	model *model = new_model( &params );

	int time = 0;
	while( one_time_step( model ) && time < params.end_time )
		time++;

	printf( "End_time:                      %i\n",  model->time );
	printf( "Total population:              %li\n", params.n_total );
	printf( "Total daily interactions:      %li\n", model->n_possible_interactions );
	printf( "Total interactions remembered: %li\n", model->n_interactions );

	/*
	printf( "\n");
	int person = 3643;
	int day = 4;
	int i;
	interaction *inter;
	printf( "%i: ", model->population[ person ].n_interactions[ day ] );
	inter = model->population[ person ].interactions[ day ];
	printf( "%li ", inter->individual->idx);
	for( i = 1; i < model->population[ person ].n_interactions[ day ]; i++ )
	{
		inter = inter->next;
		printf( "%li ", inter->individual->idx);
	}
	printf( "\n");

*/
    destroy_model( model );
 //   gsl_rng_free( rng );
    printf("Ending simulation\n");
    return 0;
}
