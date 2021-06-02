%module vaccine_utils

#ifndef SWIGR
#include <string.h>
%include <cstring.i>
%cstring_bounded_output(char* outstr, 1024);
#endif

%inline %{

short vaccine_idx( vaccine *pvaccine ) {
    return pvaccine->idx;
}

void vaccine_full_efficacy( vaccine *pvaccine, float *efficacy ) {
 	
 	short n_strains = pvaccine->n_strains;
 	
 	for( int idx = 0; idx < n_strains; idx++ )
 		efficacy[ idx ] = pvaccine->full_efficacy[ idx ];
}

void vaccine_symptoms_efficacy( vaccine *pvaccine, float *efficacy ) {

	short n_strains = pvaccine->n_strains;
 	
 	for( int idx = 0; idx < n_strains; idx++ )
 		efficacy[ idx ] = pvaccine->symptoms_efficacy[ idx ];
 }

void vaccine_severe_efficacy( vaccine *pvaccine, float *efficacy  ) {
	
	short n_strains = pvaccine->n_strains;
 	
 	for( int idx = 0; idx < n_strains; idx++ )
 		efficacy[ idx ] = pvaccine->severe_efficacy[ idx ];
}

short vaccine_time_to_protect( vaccine *pvaccine ) {
    return pvaccine->time_to_protect;
}

short vaccine_vaccine_protection_period( vaccine *pvaccine ) {
    return pvaccine->vaccine_protection_period;
}

char *vaccine_name( vaccine *pvaccine ) {
	return pvaccine->name;
}

short vaccine_n_strains( vaccine *pvaccine ) {
    return pvaccine->n_strains;
}

%}