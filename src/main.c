// main.c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

#include "model.h"
#include "network.h"
#include "params.h"
#include "utilities.h"
#include "input.h"
#include "constant.h"

#define OK       0
#define NO_INPUT 1
#define TOO_LONG 2
//FIXME steps should be configurable
#define STEPS    2

static int get_line_from_prmpt(char *prmpt, char *buff, size_t sz) {
    int ch, extra;

    // Get line with buffer overrun protection.
    if (prmpt != NULL) {
        printf ("%s", prmpt);
        fflush (stdout);
    }
    if (fgets (buff, sz, stdin) == NULL)
        return NO_INPUT;

    // If it was too long, there'll be no newline. In that case, we flush
    // to end of line so that excess doesn't affect the next call.
    if (buff[strlen(buff)-1] != '\n') {
        extra = 0;
        while (((ch = getchar()) != '\n') && (ch != EOF))
        extra = 1;
        return (extra == 1) ? TOO_LONG : OK;
    }

    // Otherwise remove newline and give string back to caller.
    buff[strlen(buff)-1] = '\0';
    return OK;
}

int main(int argc, char *argv[])
{
    printf("# Starting simulation\n");

    parameters params;	
	
    struct timespec  tv;
    double tstart, tend;

    int rc = 1;
    int steps = STEPS;
    int last_step = STEPS;
    int step = 1;
    int name_size = 0;
    char input_param_file[INPUT_CHAR_LEN];
    
    clock_gettime( CLOCK_REALTIME,&tv );
    tstart = ( tv.tv_sec ) + ( tv.tv_nsec ) / 1e9;

	printf("# Read command-line args\n");
	read_command_line_args( &params, argc, argv );
	
	printf("# Read input parameter file\n");
	read_param_file( &params );
	check_params( &params );

	printf("# Start model set-up\n");
    gsl_rng_env_setup();
    rng = gsl_rng_alloc ( gsl_rng_default );
	
	gsl_rng_set( rng, params.rng_seed );
	model *model = new_model( &params );

    while( step <= steps ) {
        printf("Step %d\n", step);
        rc = one_time_step( model );
        if (!rc) {
            printf("Error calculating results");
        }
	    printf("Time,total_infected,total_case,n_presymptom,n_asymptom,n_quarantine,n_tests,n_symptoms,n_hospital,n_death,n_recovered\n");
        printf( "%i,%li,%li,%li,%li,%li,%li,%li,%li,%li,%li\n",
				model->time,
				n_total( model, PRESYMPTOMATIC ) + n_total( model, ASYMPTOMATIC ),
				n_total( model, CASE ),
				n_current( model, PRESYMPTOMATIC ),
				n_current( model, ASYMPTOMATIC ),
				n_current( model, QUARANTINED ),
				n_daily( model, TEST_TAKE, model->time + 1),
				n_current( model, SYMPTOMATIC ),
				n_current( model, HOSPITALISED ),
				n_current( model, DEATH ),
				n_current( model, RECOVERED )
		);
        /* Send results to Python program to calculate new params */
        if (step+1 <= last_step) {
            /* 
             * FIXME: Turn on write individual file flag
             * This will create a temporary parameters file called:
             * individual_file_Run<param_line_number>.csv
             * This should not be configured form the input .csv file
             */
            int sys_write_individual = model->params->sys_write_individual;
            if (!sys_write_individual) 
                model->params->sys_write_individual = 1;

            /* Write output files */
            write_output_files(model, &params);

            /* FIXME: Turn off write individial file flag */
            if (!sys_write_individual) 
                model->params->sys_write_individual = 0;

            printf("Get new params for step %d\n", step+1);
            rc = get_line_from_prmpt("New params file>",
                                     input_param_file,
                                     sizeof(input_param_file));
            if (rc == NO_INPUT) {
                /* Extra NL since my system doesn't output that on EOF */
                printf("No input\n");
                return 1;
            }

            if (rc == TOO_LONG) {
                printf("Input too long [%s]\n", input_param_file);
                return 1;
            }

            printf ("OK [%s]\n", input_param_file);
            
            /* Load new params */
            strncpy(model->params->input_param_file,
                    input_param_file,
                    sizeof(model->params->input_param_file) - 1);
            name_size = sizeof(model->params->input_param_file);
            model->params->input_param_file[name_size - 1] = '\0';
        }
        step++;
    }

	printf( "\n# End_time:                    %i\n",  model->time );
	printf( "# Total population:              %li\n", params.n_total );
	printf( "# Total edges in network:        %li\n", model->random_network->n_edges );
	printf( "# Total total interactions:      %li\n", model->n_total_intereactions );
	printf( "# Total infected:                %li\n", n_total( model, PRESYMPTOMATIC ) + n_total( model, ASYMPTOMATIC ) );
	printf( "# Total cases:                   %li\n", n_total( model, CASE ) );
	printf( "# Total cases children:          %li\n", n_total_age( model, CASE, AGE_0_17 ) );
	printf( "# Total cases adult:             %li\n", n_total_age( model, CASE, AGE_18_64 ) );
	printf( "# Total cases elderly:           %li\n", n_total_age( model, CASE, AGE_65 ) );
	printf( "# Total deaths:                  %li\n", n_total( model, DEATH ) );
	printf( "# Total deaths children:         %li\n", n_total_age( model, DEATH, AGE_0_17 ) );
	printf( "# Total deaths adult:            %li\n", n_total_age( model, DEATH, AGE_18_64 ) );
	printf( "# Total deaths elderly:          %li\n", n_total_age( model, DEATH, AGE_65 ) );
	printf( "# Total quarantined days:        %li\n", model->n_quarantine_days );

    write_output_files( model, &params );
	
	destroy_model( model );
	gsl_rng_free( rng );

    clock_gettime( CLOCK_REALTIME, &tv );
    tend = ( tv.tv_sec ) + ( tv.tv_nsec ) / 1e9;

    printf("# Ending simulation, run time:   %.2fs\n", tend - tstart );
    return 0;
}
