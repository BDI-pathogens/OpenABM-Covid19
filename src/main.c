// main.c

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

#include "model.h"
#include "params.h"

int main(int argc, char *argv[])
{
    printf("Starting simulation\n");


    parameters params;
    params.n_total = 1e5;
    params.mean_daily_interactions = 10;
    params.days_of_interactions = 5;

	model *model = new_model( &params );

	printf( "Total population:              %li\n", params.n_total );
	printf( "Total daily interactions:      %li\n", model->n_possible_interactions );
	printf( "Total interactions remembered: %li\n", model->n_interactions );

    destroy_model( model );
    printf("Ending simulation\n");
    return 0;
}
