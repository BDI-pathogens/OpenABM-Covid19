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
#include "constant.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

int main(int argc, char *argv[])
{
    printf("Starting simulation\n");

    parameters params;	

	printf("Read command-line args\n");
	read_command_line_args(&params, argc, argv);
	
	printf("Read input parameter file\n");
	read_param_file( &params );
	check_params( &params );

	printf("Start model set-up\n");
    gsl_rng_env_setup();
    rng = gsl_rng_alloc ( gsl_rng_default);
	model *model = new_model( &params );

	while( one_time_step( model ) && model->time < params.end_time )
		printf( "Time %2i; n_infected %li; n_sypmtoms %li\n",
				model->time, model->infected.n_total, model->symptomatic.n_total );

	printf( "End_time:                      %i\n",  model->time );
	printf( "Total population:              %li\n", params.n_total );
	printf( "Total daily interactions:      %li\n", model->n_possible_interactions );
	printf( "Total interactions remembered: %li\n", model->n_interactions );
	printf( "Total infected:                %li\n", model->infected.n_total );

    destroy_model( model );
 //   gsl_rng_free( rng );
    printf("Ending simulation\n");
    return 0;
}
