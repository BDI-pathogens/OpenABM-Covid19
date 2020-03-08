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

	while( model->time < params.end_time && one_time_step( model ) )
		printf( "Time %2i; total_infected %li; n_infected %li; n_asymptom %li; n_sypmtoms %li; n_hospital %li; n_death %li; n_recovered %li\n",
				model->time,
				model->infected.n_total,
				model->infected.n_current,
				model->asymptomatic.n_current,
				model->symptomatic.n_current,
   			    model->hospitalised.n_current,
   			    model->death.n_current,
   			    model->recovered.n_current
		);

	printf( "\nEnd_time:                      %i\n",  model->time );
	printf( "Total population:              %li\n", params.n_total );
	printf( "Total total interactions:      %li\n", model->n_total_intereactions );
	printf( "Total infected:                %li\n", model->infected.n_total );

    destroy_model( model );
 //   gsl_rng_free( rng );
    printf("Ending simulation\n");
    return 0;
}
