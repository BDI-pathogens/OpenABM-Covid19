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

short vaccine_vaccine_type( vaccine *pvaccine ) {
    return pvaccine->vaccine_type[ 0 ];
}

float vaccine_efficacy( vaccine *pvaccine ) {
    return pvaccine->efficacy[ 0 ];
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

%}