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

static int get_line_from_prmpt (char *prmpt, char *buff, size_t sz) {
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

	printf("Time,total_infected,total_case,n_presymptom,n_asymptom,n_quarantine,n_tests,n_symptoms,n_hospital,n_death,n_recovered\n");
    int rc = 1;
    int steps = 10;
    int last_step = 8;
    int step = 1;
    char input_param_file[INPUT_CHAR_LEN];
    char output_file_dir[INPUT_CHAR_LEN];

    /* Store output file directory */
    strncpy(output_file_dir, model->params->output_file_dir, sizeof(output_file_dir) - 1);
    output_file_dir[sizeof(output_file_dir) - 1] = '\0'; 
    
    while( rc && step <= steps ) {
        printf("Step %d\n", step);
        rc = one_time_step( model );
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
            /* Write temporary results to file */
            char tmp_file_dir[20];
            snprintf(tmp_file_dir, sizeof(tmp_file_dir)-1, "temp_output_%d", step+1);
            strncpy(model->params->output_file_dir, tmp_file_dir, sizeof(model->params->output_file_dir) - 1);
            model->params->output_file_dir[sizeof(model->params->output_file_dir) - 1] = '\0'; 
            write_output_files(model, &params);

            printf("Get new params for step %d\n", step+1);
            rc = get_line_from_prmpt ("New params file>", input_param_file, sizeof(input_param_file));
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
            strncpy(model->params->input_param_file, input_param_file, sizeof(model->params->input_param_file) - 1);
            model->params->input_param_file[sizeof(model->params->input_param_file) - 1] = '\0'; 
        }
        step++;
    }
	printf( "\n# End_time:                      %i\n",  model->time );
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

    /* Set original output file name */
    strncpy(model->params->output_file_dir, output_file_dir, sizeof(model->params->output_file_dir) - 1);
    model->params->output_file_dir[sizeof(model->params->output_file_dir) - 1] = '\0'; 
    
    write_output_files( model, &params );
	
	destroy_model( model );
	gsl_rng_free( rng );

    clock_gettime( CLOCK_REALTIME, &tv );
    tend = ( tv.tv_sec ) + ( tv.tv_nsec ) / 1e9;

    printf("# Ending simulation, run time:   %.2fs\n", tend - tstart );
    return 0;
}
