// main.c

#include <stdio.h>
#include <time.h>
#include <math.h>

#include "model.h"
#include "network.h"
#include "params.h"
#include "input.h"
#include "constant.h"

int main(int argc, char *argv[])
{
    printf("# Starting simulation\n");

    parameters params;	

    struct timespec  tv;
    double tstart, tend;
    int idx;
	char date_time[30];
	
    clock_gettime( CLOCK_REALTIME,&tv);
    tstart = ( tv.tv_sec ) + ( tv.tv_nsec ) / 1e9;

	time_t time_now = time( NULL );
	strftime(date_time, sizeof(date_time), "# Date: %d-%m-%Y %I:%M:%S", localtime(&time_now)); 
	puts(date_time);
	
	printf("# Read command-line args\n");
	read_command_line_args(&params, argc, argv);
	
	printf("# Read input parameter file\n");
	read_param_file( &params );
	check_params( &params );
	
	printf("# Read household demographics file\n");
	read_household_demographics_file( &params );
	
    printf("# Read hospital parameter file\n");
    read_hospital_param_file( &params );

	printf("# Start model set-up\n");

    model *model = new_model( &params );
	
	printf("# param_id: %li\n", params.param_id);
	printf("# rng_seed: %li\n", params.rng_seed);
	printf("# param_line_number: %d\n", params.param_line_number);
	
//	if( params.sys_write_individual  )
//		write_trace_tokens_ts( model, TRUE );

	printf( "time,lockdown,test_on_symptoms,app_on,total_infected,total_case,n_presymptom,n_asymptom,n_quarantine,n_tests,n_symptoms,n_hospital,n_critical,n_death,n_recovered,n_waiting,n_general,n_icu,n_discharged,n_mortuary\n");
	while( model->time < params.end_time && one_time_step( model ) )
	{
		printf( "%i,%i,%i,%i,%li,%li,%li,%li,%li,%li,%li,%li,%li,%li,%li,%li,%li,%li,%li,%li\n",
				model->time,
				params.lockdown_on,
				params.test_on_symptoms,
				params.app_turned_on,
				n_total( model, PRESYMPTOMATIC ) + n_total( model, PRESYMPTOMATIC_MILD ) + n_total( model, ASYMPTOMATIC ),
				n_total( model, CASE ),
				n_current( model, PRESYMPTOMATIC ) + n_current( model, PRESYMPTOMATIC_MILD ),
				n_current( model, ASYMPTOMATIC ),
				n_current( model, QUARANTINED ),
				n_daily( model, TEST_RESULT, model->time + 1),
				n_current( model, SYMPTOMATIC ),
				n_current( model, HOSPITALISED ),
				n_current( model, CRITICAL ),
				n_current( model, DEATH ),
				n_current( model, RECOVERED ),
				n_current( model, WAITING ),
				n_current( model, GENERAL ),
				n_current( model, ICU),
				n_current( model, DISCHARGED),
				n_current( model, MORTUARY)
		);
//		if( params.sys_write_individual )
//			write_trace_tokens_ts( model, FALSE );

	};

	printf( "\n# End_time:                      %i\n",  model->time );
	printf( "# Total population:              %li\n", params.n_total );
	printf( "# Total edges in network:        %li\n", model->random_network->n_edges );
	printf( "# Total total interactions:      %li\n", model->n_total_intereactions );
	printf( "# Total infected:                %li\n", n_total( model, PRESYMPTOMATIC ) + n_total( model, PRESYMPTOMATIC_MILD ) + n_total( model, ASYMPTOMATIC ) );
	printf( "# Total cases:                   %li\n", n_total( model, CASE ) );
	for( idx = 0; idx < N_AGE_GROUPS; idx++ )
		printf( "# Total cases %11s:       %li\n", AGE_TEXT_MAP[idx], n_total_age( model, CASE, idx ) );
	for( idx = 0; idx < N_AGE_GROUPS; idx++ )
		printf( "# Total deaths %11s:      %li\n", AGE_TEXT_MAP[idx], n_total_age( model, DEATH, idx ) );
	printf( "# Total quarantined days:        %li\n", model->n_quarantine_days );

	write_output_files( model, &params );

	destroy_model( model );
	destroy_params( &params );

    clock_gettime( CLOCK_REALTIME, &tv );
    tend = ( tv.tv_sec ) + ( tv.tv_nsec ) / 1e9;

    printf("# Ending simulation, run time:   %.2fs\n", tend - tstart );
    return 0;
}
