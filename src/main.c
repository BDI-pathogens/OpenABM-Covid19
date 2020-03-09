// main.c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

#include "model.h"
#include "params.h"
#include "utilities.h"
#include "input.h"
#include "constant.h"

int main(int argc, char *argv[])
{
    printf("# Starting simulation\n");

    parameters params;	
    struct timespec  tv;
    double tstart, tend;

    clock_gettime( CLOCK_REALTIME,&tv);
    tstart = ( tv.tv_sec ) + ( tv.tv_nsec ) / 1e9;

	printf("# Read command-line args\n");
	read_command_line_args(&params, argc, argv);
	
	printf("# Read input parameter file\n");
	read_param_file( &params );
	check_params( &params );

	printf("# Start model set-up\n");
    gsl_rng_env_setup();
    rng = gsl_rng_alloc ( gsl_rng_default);
	model *model = new_model( &params );

	printf( "Time,\ttotal_infected,\tn_presymptom,\tn_asymptom, \tn_quarantine, \tn_symptoms,\tn_hospital,\tn_death,\tn_recovered\n");
	while( model->time < params.end_time && one_time_step( model ) )
		printf( "%2i,\t%li,\t%li,\t%li,\t%li,\t%li,\t%li,\t%li,\t%li\n",
				model->time,
				model->presymptomatic.n_total + model->asymptomatic.n_total,
				model->presymptomatic.n_current,
				model->asymptomatic.n_current,
				model->quarantined.n_current,
				model->symptomatic.n_current,
   			    model->hospitalised.n_current,
   			    model->death.n_current,
   			    model->recovered.n_current
		);
	printf( "\n# End_time:                      %i\n",  model->time );
	printf( "# Total population:              %li\n", params.n_total );
	printf( "# Total total interactions:      %li\n", model->n_total_intereactions );
	printf( "# Total infected:                %li\n", model->presymptomatic.n_total + model->asymptomatic.n_total);

    destroy_model( model );
 //   gsl_rng_free( rng );

    clock_gettime( CLOCK_REALTIME, &tv );
    tend = ( tv.tv_sec ) + ( tv.tv_nsec ) / 1e9;

    printf("# Ending simulation, run time:   %.2fs\n", tend - tstart );
    return 0;
}
